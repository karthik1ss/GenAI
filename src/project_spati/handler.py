import json
import os
import re
from http import HTTPStatus
from textwrap import dedent
from typing import Optional

import boto3
from aws_lambda_powertools import Logger, Metrics, single_metric
from aws_lambda_powertools.event_handler import APIGatewayRestResolver, Response, content_types
from aws_lambda_powertools.event_handler.exceptions import BadRequestError
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.utilities.typing import LambdaContext
from mypy_boto3_bedrock_agent_runtime import AgentsforBedrockRuntimeClient
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from mypy_boto3_bedrock_runtime.literals import GuardrailContentSourceType
from pydantic import BaseModel, Field, ValidationError

from .exceptions import (
    GuardrailInterventionError,
    NoAnswerFromModelError,
    NoContextFoundError,
    UnableToProvideAnswerError,
)


class UserInput(BaseModel):
    question: str = Field(
        min_length=1,
        # Default `Characters in query text` quota of Amazon Kendra is 1,000
        max_length=1000,
    )


SYSTEM_PROMPT = dedent(
    """\
<General Instructions>
You are an AWS Cloud Practitioner expert.
Your role is to provide accurate and helpful information related to AWS services \
and products based on the given context.
</General Instructions>

<Task>
Read the user's question carefully, which is enclosed within <question> tags. \
Analyze the contextual information provided within <context> tags, \
which may contain multiple <document> sections with relevant details. \
Your task is to utilize this context to formulate a comprehensive answer to the \
question and provide it within <answer> tags. Do not repeat the context. \
If you do not have sufficient information to answer the question, simply respond with \
"I do not have enough information to answer."
</Task>
"""
)

app = APIGatewayRestResolver()
logger = Logger(service="ProjectSPATI")
metrics = Metrics(namespace="ProjectSPATI")
metrics.set_default_dimensions(service="ApiHandler")

MODEL_ID = os.environ["MODEL_ID"]
KNOWLEDGE_BASE_ID = os.environ["KNOWLEDGE_BASE_ID"]
GUARDRAIL_ID = os.environ["GUARDRAIL_ID"]
GUARDRAIL_VERSION = os.environ["GUARDRAIL_VERSION"]
AWS_REGION = os.environ["AWS_REGION"]

bedrock_agent_runtime: AgentsforBedrockRuntimeClient = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)  # type: ignore
bedrock_runtime: BedrockRuntimeClient = boto3.client("bedrock-runtime", region_name=AWS_REGION)  # type: ignore


def apply_guardrail(
    bedrock_runtime: BedrockRuntimeClient,
    content: str,
    source: GuardrailContentSourceType,
) -> None:
    """Guard the content with Bedrock Guardrail. If the content is not flagged by Guardrail,
    forward it to the next step in the flow.
    """
    result = bedrock_runtime.apply_guardrail(
        guardrailIdentifier=GUARDRAIL_ID,
        guardrailVersion=GUARDRAIL_VERSION,
        source=source,
        content=[
            {
                "text": {
                    "text": content,
                    "qualifiers": [
                        "guard_content",
                    ],
                }
            },
        ],
    )

    # Emit guardrail intervention metric with dimension "source" as INPUT or OUTPUT
    with single_metric(
        name="GuardrailIntervened",
        unit=MetricUnit.Count,
        value=int(result["action"] != "NONE"),
        namespace=metrics.namespace,
        default_dimensions=metrics.default_dimensions,
    ) as guardrail_metric:
        guardrail_metric.add_dimension(name="source", value=source)

    if result["action"] != "NONE":
        logger.warning(
            f"Guardrail ({GUARDRAIL_ID}) intervened ({result['ResponseMetadata']['RequestId']})"
        )
        raise GuardrailInterventionError(source, result.get("assessments", []))


def retrieve_context(
    bedrock_agent_runtime: AgentsforBedrockRuntimeClient, query: str
) -> list[dict]:
    """Retrieve relevant documents from the knowledge base."""
    retrieve_response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 10}},
    )

    retrieved_documents = []
    for result in retrieve_response.get("retrievalResults", []):
        content = result["content"]
        if "text" in content:
            retrieved_documents.append({"text": content["text"]})
        else:
            raise RuntimeError(
                "Unexpected content field in the Retrieve response "
                + f"({retrieve_response['ResponseMetadata']['RequestId']})"
            )

    if not retrieved_documents:
        raise NoContextFoundError()

    return retrieved_documents


def format_context(documents: list[str]) -> str:
    """Format retrieved documents into a context string."""
    context_list = []
    for document in documents:
        context_list.append(f"<document>{document}</document>")
    return dedent(
        f"""\
    <context>
    {"".join(context_list)}
    </context>
    """
    )


def format_query(question: str) -> str:
    """Format context and question into a query string."""
    return dedent(
        f"""\
    <question>
    {question}
    </question>
    """
    )


def inference(bedrock_runtime: BedrockRuntimeClient, system_prompt: str, user_prompt: str) -> str:
    """Invoke LLM and return the response."""
    converse_response = bedrock_runtime.converse(
        modelId=MODEL_ID,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={"maxTokens": 500, "temperature": 0},
    )

    output = converse_response["output"]
    if (
        "message" not in output
        or not output["message"]["content"]
        or "text" not in output["message"]["content"][0]
    ):
        raise RuntimeError(
            "Unexpected output field in the Converse response "
            + f"({converse_response['ResponseMetadata']['RequestId']})"
        )

    llm_output = output["message"]["content"][0]["text"]

    try:
        token_usage = converse_response["usage"]
        metrics.add_metric(
            name="InputTokens", unit=MetricUnit.Count, value=token_usage["inputTokens"]
        )
        metrics.add_metric(
            name="OutputTokens",
            unit=MetricUnit.Count,
            value=token_usage["outputTokens"],
        )
        metrics.add_metric(
            name="TotalTokens", unit=MetricUnit.Count, value=token_usage["totalTokens"]
        )
    except KeyError:
        raise RuntimeError(
            "No token usage information found in the Converse response "
            + f"({converse_response['ResponseMetadata']['RequestId']})"
        )

    return llm_output


def parse_output(text: str) -> Optional[str]:
    """Extract answer from LLM output."""
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()

    return None


def construct_user_prompt(documents: list[str], question: str) -> str:
    """Prepare user prompt."""
    context = format_context(documents)
    query = format_query(question)
    return context + "\n" + query


def retrieve_and_generate(
    bedrock_agent_runtime: AgentsforBedrockRuntimeClient,
    bedrock_runtime: BedrockRuntimeClient,
    question: str,
) -> dict:
    """Orchestrate the retrieve and generate process."""

    # Step 1: Apply guardrail on input
    apply_guardrail(bedrock_runtime, content=question, source="INPUT")

    # Step 2: Retrieve context
    context = retrieve_context(bedrock_agent_runtime, question)

    # Step 3: Prepare user prompt
    user_prompt = construct_user_prompt([c["text"] for c in context], question)

    # Step 4: Generate answer
    llm_output = inference(bedrock_runtime, SYSTEM_PROMPT, user_prompt)

    # Step 5: Extract answer from LLM output
    answer = parse_output(llm_output)

    # Step 6: Apply guardrail on output
    if answer:
        apply_guardrail(bedrock_runtime, content=answer, source="OUTPUT")
        return {"answer": answer}

    # Validate the raw output with guardrail before raising the exception
    apply_guardrail(bedrock_runtime, content=llm_output, source="OUTPUT")
    raise NoAnswerFromModelError(llm_output)


@app.post("/")
def query_handler() -> dict:
    post_data: dict = app.current_event.json_body

    # Validate user input
    try:
        user_input = UserInput(**post_data)
    except ValidationError as exception:
        # Parse the pydantic error to make the error message in response easier to read
        errors = [f'Field {e["loc"][-1]}: {e["msg"]}' for e in exception.errors()]
        raise BadRequestError(f"Input validation failed. Errors: {errors}")

    # Retrieve context and invoke LLM to inference
    return retrieve_and_generate(bedrock_agent_runtime, bedrock_runtime, user_input.question)


@app.exception_handler(UnableToProvideAnswerError)
def handle_rag_error(ex: UnableToProvideAnswerError):
    logger.warning("Unable to provide answer", exc_info=ex)
    error_name = type(ex).__name__
    metrics.add_metric(name=error_name, unit="Count", value=1)
    response_body = {"error": {"code": error_name, "message": ex.error_message()}}
    if isinstance(ex, NoAnswerFromModelError):
        response_body["error"]["model_response"] = ex.llm_output
    return Response(
        status_code=HTTPStatus.OK,
        content_type=content_types.APPLICATION_JSON,
        body=json.dumps(response_body),
    )


@app.exception_handler(BadRequestError)
def handle_client_error(ex: BadRequestError):
    logger.warning("Bad request", exc_info=ex)
    metrics.add_metric(name="BadRequestError", unit="Count", value=1)
    return Response(
        status_code=HTTPStatus.BAD_REQUEST,
        content_type=content_types.APPLICATION_JSON,
        body=json.dumps({"error": {"code": "BadRequestError", "message": ex.msg}}),
    )


@app.exception_handler(Exception)
def handle_server_error(ex: Exception):
    logger.error("Internal error", exc_info=ex)
    metrics.add_metric(name="InternalError", unit="Count", value=1)
    return Response(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        content_type=content_types.APPLICATION_JSON,
        body=json.dumps({"error": {"code": "InternalError", "message": "Internal error."}}),
    )


@metrics.log_metrics
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    return app.resolve(event, context)
