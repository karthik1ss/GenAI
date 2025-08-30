import json
import logging
import os
from http import HTTPStatus
from textwrap import dedent
from unittest.mock import ANY, MagicMock, call, patch

import pytest

from project_spati import lambda_handler
from project_spati.exceptions import (
    GuardrailInterventionError,
    NoAnswerFromModelError,
    NoContextFoundError,
)
from project_spati.handler import (
    SYSTEM_PROMPT,
    format_context,
    format_query,
    inference,
    parse_output,
    retrieve_context,
)

MODEL_ID = os.environ["MODEL_ID"]
KNOWLEDGE_BASE_ID = os.environ["KNOWLEDGE_BASE_ID"]
GUARDRAIL_ID = os.environ["GUARDRAIL_ID"]
GUARDRAIL_VERSION = os.environ["GUARDRAIL_VERSION"]


@pytest.fixture
def mock_bedrock_agent_runtime():
    yield MagicMock()


@pytest.fixture
def mock_bedrock_runtime():
    yield MagicMock()


@pytest.fixture
def mock_event():
    yield {
        "resource": "/",
        "path": "/",
        "httpMethod": "POST",
        "multiValueHeaders": {"Content-Type": ["application/json"]},
        "multiValueQueryStringParameters": None,
        "pathParameters": None,
        "stageVariables": None,
        "requestContext": {
            "accountId": "12345678912",
            "apiId": "fake_api_id",
            "httpMethod": "POST",
            "identity": {
                "accessKey": "fake_access_key",
                "accountId": "fake_account_id",
                "caller": "fake_caller",
                "user": "fake_user",
                "userAgent": "fake_agent",
                "userArn": "arn:aws:sts::12345678912:assumed-role/BONESBootstrapHydra-MyTestLambda/CODETEST_MyTestLambda_0678761137",
                "sourceIp": "0.0.0.0",
            },
            "path": "/",
            "requestId": "fake_request_id",
            "resourceId": "fake_resource_id",
            "resourcePath": "/",
            "stage": "default",
        },
        "body": "",
        "isBase64Encoded": False,
    }


@pytest.fixture
def mock_retrieve_response():
    yield {
        "ResponseMetadata": {
            "RequestId": "fake-request-id",
            "HTTPStatusCode": 200,
            "RetryAttempts": 0,
        },
        "retrievalResults": [
            {
                "content": {
                    "text": "Content Foo",
                    "type": "TEXT",
                },
                "location": {
                    "kendraDocumentLocation": {
                        "uri": "https://s3.us-west-2.amazonaws.com/fake-bucket/rag/lambda-developer-guide-231030/foo.md"
                    },
                    "type": "KENDRA",
                },
                "metadata": {
                    "x-amz-kendra-passage-id": "passage-id-foo",
                    "x-amz-kendra-document-id": "fake.foo",
                    "x-amz-kendra-score-confidence": "HIGH",
                    "x-amz-kendra-document-title": "Foo",
                },
                "score": 0.75,
            },
            {
                "content": {
                    "text": "Content Bar",
                    "type": "TEXT",
                },
                "location": {
                    "kendraDocumentLocation": {
                        "uri": "https://s3.us-west-2.amazonaws.com/fake-bucket/rag/blogs/bar.md"
                    },
                    "type": "KENDRA",
                },
                "metadata": {
                    "x-amz-kendra-passage-id": "passage-id-bar",
                    "x-amz-kendra-document-id": "fake.bar",
                    "x-amz-kendra-score-confidence": "HIGH",
                    "x-amz-kendra-document-title": "Bar",
                },
                "score": 0.65,
            },
        ],
    }


@pytest.fixture
def mock_non_textual_retrieve_response():
    yield {
        "ResponseMetadata": {
            "RequestId": "fake-request-id",
            "HTTPStatusCode": 200,
            "RetryAttempts": 0,
        },
        "retrievalResults": [
            {
                "content": {
                    "byteContent": "data:image/jpeg;base64,${123456}",
                    "type": "IMAGE",
                },
                "location": {
                    "kendraDocumentLocation": {
                        "uri": "https://s3.us-west-2.amazonaws.com/fake-bucket/rag/lambda-developer-guide-231030/foo.jpeg"
                    },
                    "type": "KENDRA",
                },
                "metadata": {
                    "x-amz-kendra-passage-id": "passage-id-foo",
                    "x-amz-kendra-document-id": "fake.foo",
                    "x-amz-kendra-score-confidence": "HIGH",
                    "x-amz-kendra-document-title": "Foo",
                },
                "score": 0.5,
            },
        ],
    }


@pytest.fixture
def mock_empty_retrieve_response():
    yield {"retrievalResults": []}


@pytest.fixture
def mock_converse_response():
    yield {
        "ResponseMetadata": {
            "RequestId": "fake-request-id",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {
                "date": "Wed, 30 Apr 2025 23:28:34 GMT",
                "content-type": "application/json",
                "content-length": "509",
                "connection": "keep-alive",
                "x-amzn-requestid": "fake-request-id",
            },
            "RetryAttempts": 0,
        },
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "<answer>fake answer</answer>"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 200, "outputTokens": 100, "totalTokens": 300},
        "metrics": {"latencyMs": 1000},
    }


@pytest.fixture
def mock_converse_no_token_usage_response():
    yield {
        "ResponseMetadata": {
            "RequestId": "fake-request-id",
            "HTTPStatusCode": 200,
            "RetryAttempts": 0,
        },
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "<answer>fake answer</answer>"}],
            }
        },
    }


@pytest.fixture
def mock_converse_no_answer_response():
    yield {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Sorry I'm not able to provide answer."}],
            }
        },
        "usage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120},
    }


@pytest.fixture
def mock_converse_no_message_response():
    yield {
        "ResponseMetadata": {
            "RequestId": "fake-request-id",
            "HTTPStatusCode": 200,
        },
        "output": {},
    }


@pytest.fixture
def mock_apply_guardrail_pass_response():
    yield {
        "action": "NONE",
    }


@pytest.fixture
def mock_apply_guardrail_block_response():
    yield {
        "action": "GUARDRAIL_INTERVENED",
        "ResponseMetadata": {"RequestId": "mock-request-id"},
        "assessments": [
            {
                "contentPolicy": {
                    "filters": [
                        {
                            "type": "PROMPT_ATTACK",
                            "confidence": "HIGH",
                            "filterStrength": "HIGH",
                            "action": "BLOCKED",
                        }
                    ]
                },
                "invocationMetrics": {
                    "guardrailProcessingLatency": 280,
                    "usage": {
                        "topicPolicyUnits": 0,
                        "contentPolicyUnits": 1,
                        "wordPolicyUnits": 0,
                        "sensitiveInformationPolicyUnits": 1,
                        "sensitiveInformationPolicyFreeUnits": 0,
                        "contextualGroundingPolicyUnits": 0,
                    },
                    "guardrailCoverage": {"textCharacters": {"guarded": 140, "total": 140}},
                },
            }
        ],
    }


def capture_metrics_output_multiple_emf_objects(capsys):
    """Parse EMF metrics from std out"""
    return [
        json.loads(line.strip()) for line in capsys.readouterr().out.split("\n") if "_aws" in line
    ]


def expected_guardrail_intervention_metrics(source, value):
    return {
        "_aws": {
            "Timestamp": ANY,
            "CloudWatchMetrics": [
                {
                    "Namespace": "ProjectSPATI",
                    "Dimensions": [["service", "source"]],
                    "Metrics": [{"Name": "GuardrailIntervened", "Unit": "Count"}],
                }
            ],
        },
        "service": "ApiHandler",
        "source": source,
        "GuardrailIntervened": [value],
    }


def assert_metrics_existed(actual_metric, expected_metric_count_dict):
    all_metric_names = [m["Name"] for m in actual_metric["_aws"]["CloudWatchMetrics"][0]["Metrics"]]
    for k, v in expected_metric_count_dict.items():
        assert k in actual_metric
        assert actual_metric[k] == [v]
        assert k in all_metric_names


@patch("project_spati.handler.bedrock_runtime")
@patch("project_spati.handler.bedrock_agent_runtime")
def test_retrieve_and_generate(
    mock_bedrock_agent_runtime,
    mock_bedrock_runtime,
    mock_event,
    mock_apply_guardrail_pass_response,
    mock_retrieve_response,
    mock_converse_response,
    capsys,
):
    mock_event["body"] = '{"question": "fake question"}'
    mock_bedrock_runtime.apply_guardrail.return_value = mock_apply_guardrail_pass_response
    mock_bedrock_runtime.converse.return_value = mock_converse_response
    mock_bedrock_agent_runtime.retrieve.return_value = mock_retrieve_response

    response = lambda_handler(mock_event, None)

    assert response.get("statusCode") == HTTPStatus.OK
    assert response.get("body") == '{"answer":"fake answer"}'

    # Verify apply_guardrail calls
    assert mock_bedrock_runtime.apply_guardrail.call_args_list == [
        call(
            guardrailIdentifier=GUARDRAIL_ID,
            guardrailVersion=GUARDRAIL_VERSION,
            source="INPUT",
            content=[{"text": {"text": "fake question", "qualifiers": ["guard_content"]}}],
        ),
        call(
            guardrailIdentifier=GUARDRAIL_ID,
            guardrailVersion=GUARDRAIL_VERSION,
            source="OUTPUT",
            content=[{"text": {"text": "fake answer", "qualifiers": ["guard_content"]}}],
        ),
    ]

    # Verify retrieve call
    assert mock_bedrock_agent_runtime.retrieve.call_args == call(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": "fake question"},
        retrievalConfiguration=ANY,
    )

    # Verify converse call
    assert mock_bedrock_runtime.converse.call_args == call(
        modelId=MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": ANY}]}],
        inferenceConfig=ANY,
    )

    # Verify metrics
    # When no guardrail intervention happened, guardrail intervention metric values should be 0 and
    # token consumption metrics should be emitted.
    metrics = capture_metrics_output_multiple_emf_objects(capsys)
    assert metrics[0] == expected_guardrail_intervention_metrics("INPUT", 0)
    assert metrics[1] == expected_guardrail_intervention_metrics("OUTPUT", 0)
    assert_metrics_existed(
        metrics[2],
        {
            "InputTokens": 200,
            "OutputTokens": 100,
            "TotalTokens": 300,
        },
    )


@patch("project_spati.handler.bedrock_runtime")
@patch("project_spati.handler.bedrock_agent_runtime")
def test_converse_no_token_usage(
    mock_bedrock_agent_runtime,
    mock_bedrock_runtime,
    mock_event,
    mock_apply_guardrail_pass_response,
    mock_retrieve_response,
    mock_converse_no_token_usage_response,
    capsys,
):
    mock_event["body"] = '{"question": "fake question"}'
    mock_bedrock_runtime.apply_guardrail.return_value = mock_apply_guardrail_pass_response
    mock_bedrock_runtime.converse.return_value = mock_converse_no_token_usage_response
    mock_bedrock_agent_runtime.retrieve.return_value = mock_retrieve_response

    response = lambda_handler(mock_event, None)

    assert response.get("statusCode") == HTTPStatus.INTERNAL_SERVER_ERROR

    actual_response_json = json.loads(response.get("body"))
    assert "message" not in actual_response_json
    assert "error" in actual_response_json
    assert actual_response_json["error"]["code"] == "InternalError"
    assert actual_response_json["error"]["message"] == "Internal error."

    metrics = capture_metrics_output_multiple_emf_objects(capsys)
    assert metrics[0] == expected_guardrail_intervention_metrics("INPUT", 0)
    assert_metrics_existed(
        metrics[1],
        {
            "InternalError": 1,
        },
    )


@patch("project_spati.handler.bedrock_runtime")
@patch("project_spati.handler.bedrock_agent_runtime")
def test_converse_no_message(
    mock_bedrock_agent_runtime,
    mock_bedrock_runtime,
    mock_event,
    mock_apply_guardrail_pass_response,
    mock_retrieve_response,
    mock_converse_no_message_response,
    capsys,
):
    mock_event["body"] = '{"question": "fake question"}'
    mock_bedrock_runtime.apply_guardrail.return_value = mock_apply_guardrail_pass_response
    mock_bedrock_runtime.converse.return_value = mock_converse_no_message_response
    mock_bedrock_agent_runtime.retrieve.return_value = mock_retrieve_response

    response = lambda_handler(mock_event, None)

    assert response.get("statusCode") == HTTPStatus.INTERNAL_SERVER_ERROR

    actual_response_json = json.loads(response.get("body"))
    assert "message" not in actual_response_json
    assert "error" in actual_response_json
    assert actual_response_json["error"]["code"] == "InternalError"
    assert actual_response_json["error"]["message"] == "Internal error."

    metrics = capture_metrics_output_multiple_emf_objects(capsys)
    assert metrics[0] == expected_guardrail_intervention_metrics("INPUT", 0)
    assert_metrics_existed(
        metrics[1],
        {
            "InternalError": 1,
        },
    )


@patch("project_spati.handler.bedrock_runtime")
@patch("project_spati.handler.bedrock_agent_runtime")
def test_guardrailed_question(
    mock_bedrock_agent_runtime,
    mock_bedrock_runtime,
    mock_event,
    mock_apply_guardrail_block_response,
    capsys,
    caplog,
):
    mock_event["body"] = '{"question": "fake question"}'
    mock_bedrock_runtime.apply_guardrail.return_value = mock_apply_guardrail_block_response

    caplog.set_level(logging.WARNING)

    response = lambda_handler(mock_event, None)

    assert caplog.records[-1].message == "Unable to provide answer"
    assert isinstance(caplog.records[-1].exc_info[1], GuardrailInterventionError)

    mock_bedrock_runtime.apply_guardrail.assert_called_once()
    mock_bedrock_agent_runtime.retrieve.assert_not_called()
    mock_bedrock_runtime.converse.assert_not_called()
    assert response.get("statusCode") == HTTPStatus.OK

    actual_response_json = json.loads(response.get("body"))
    assert "message" not in actual_response_json
    assert "error" in actual_response_json
    assert actual_response_json["error"]["code"] == "GuardrailInterventionError"
    assert (
        actual_response_json["error"]["message"] == "Guardrail intervened on INPUT. "
        'Assessment: [{"contentPolicy": {"filters": [{"type": "PROMPT_ATTACK", '
        '"confidence": "HIGH", "filterStrength": "HIGH", "action": "BLOCKED"}]}}]'
    )

    # When LLM input is intervened by guardrail, there should only be one metric emitted because
    # other steps in chain won't be executed.
    metrics = capture_metrics_output_multiple_emf_objects(capsys)
    assert metrics[0] == expected_guardrail_intervention_metrics("INPUT", 1)
    assert_metrics_existed(
        metrics[1],
        {
            "GuardrailInterventionError": 1,
        },
    )


@patch("project_spati.handler.bedrock_runtime")
@patch("project_spati.handler.bedrock_agent_runtime")
def test_guardrailed_answer(
    mock_bedrock_agent_runtime,
    mock_bedrock_runtime,
    mock_event,
    mock_apply_guardrail_pass_response,
    mock_apply_guardrail_block_response,
    mock_retrieve_response,
    mock_converse_response,
    capsys,
    caplog,
):
    mock_event["body"] = '{"question": "fake question"}'
    mock_apply_guardrail_block_response["assessments"] = []
    mock_bedrock_runtime.apply_guardrail.side_effect = [
        mock_apply_guardrail_pass_response,
        mock_apply_guardrail_block_response,
    ]
    mock_bedrock_agent_runtime.retrieve.return_value = mock_retrieve_response
    mock_bedrock_runtime.converse.return_value = mock_converse_response

    caplog.set_level(logging.WARNING)

    response = lambda_handler(mock_event, None)

    assert caplog.records[-1].message == "Unable to provide answer"
    assert isinstance(caplog.records[-1].exc_info[1], GuardrailInterventionError)

    assert mock_bedrock_runtime.apply_guardrail.call_count == 2
    mock_bedrock_agent_runtime.retrieve.assert_called()
    mock_bedrock_runtime.converse.assert_called()
    assert response.get("statusCode") == HTTPStatus.OK

    actual_response_json = json.loads(response.get("body"))
    assert "message" not in actual_response_json
    assert "error" in actual_response_json
    assert actual_response_json["error"]["code"] == "GuardrailInterventionError"
    assert (
        actual_response_json["error"]["message"] == "Guardrail intervened on OUTPUT. Assessment: []"
    )

    # When LLM output is intervened by guardrail, the LLM token consumption metrics should be
    # emitted. Guardrail INPUT intervention metric value should be 0. but OUTPUT intervention metric
    # should be 1.
    metrics = capture_metrics_output_multiple_emf_objects(capsys)
    assert metrics[0] == expected_guardrail_intervention_metrics("INPUT", 0)
    assert metrics[1] == expected_guardrail_intervention_metrics("OUTPUT", 1)
    assert_metrics_existed(
        metrics[2],
        {
            "GuardrailInterventionError": 1,
            "InputTokens": 200,
            "OutputTokens": 100,
            "TotalTokens": 300,
        },
    )


@patch("project_spati.handler.bedrock_runtime")
@patch("project_spati.handler.bedrock_agent_runtime")
def test_knowledge_base_no_information(
    mock_bedrock_agent_runtime,
    mock_bedrock_runtime,
    mock_event,
    mock_apply_guardrail_pass_response,
    mock_empty_retrieve_response,
    capsys,
    caplog,
):
    mock_event["body"] = '{"question": "fake question"}'
    mock_bedrock_runtime.apply_guardrail.return_value = mock_apply_guardrail_pass_response
    mock_bedrock_agent_runtime.retrieve.return_value = mock_empty_retrieve_response

    caplog.set_level(logging.WARNING)

    response = lambda_handler(mock_event, None)

    assert caplog.records[-1].message == "Unable to provide answer"
    assert isinstance(caplog.records[-1].exc_info[1], NoContextFoundError)

    assert response.get("statusCode") == HTTPStatus.OK

    actual_response_json = json.loads(response.get("body"))
    assert "message" not in actual_response_json
    assert "error" in actual_response_json
    assert actual_response_json["error"]["code"] == "NoContextFoundError"
    assert actual_response_json["error"]["message"] == "No context found for user query."

    metrics = capture_metrics_output_multiple_emf_objects(capsys)
    assert metrics[0] == expected_guardrail_intervention_metrics("INPUT", 0)
    assert_metrics_existed(
        metrics[1],
        {
            "NoContextFoundError": 1,
        },
    )


@patch("project_spati.handler.bedrock_runtime")
@patch("project_spati.handler.bedrock_agent_runtime")
def test_knowledge_base_non_textual(
    mock_bedrock_agent_runtime,
    mock_bedrock_runtime,
    mock_event,
    mock_apply_guardrail_pass_response,
    mock_non_textual_retrieve_response,
    capsys,
):
    mock_event["body"] = '{"question": "fake question"}'
    mock_bedrock_runtime.apply_guardrail.return_value = mock_apply_guardrail_pass_response
    mock_bedrock_agent_runtime.retrieve.return_value = mock_non_textual_retrieve_response

    response = lambda_handler(mock_event, None)

    assert response.get("statusCode") == HTTPStatus.INTERNAL_SERVER_ERROR

    actual_response_json = json.loads(response.get("body"))
    assert "message" not in actual_response_json
    assert "error" in actual_response_json
    assert actual_response_json["error"]["code"] == "InternalError"
    assert actual_response_json["error"]["message"] == "Internal error."

    metrics = capture_metrics_output_multiple_emf_objects(capsys)
    assert metrics[0] == expected_guardrail_intervention_metrics("INPUT", 0)
    assert_metrics_existed(
        metrics[1],
        {
            "InternalError": 1,
        },
    )


@patch("project_spati.handler.bedrock_runtime")
@patch("project_spati.handler.bedrock_agent_runtime")
def test_llm_no_answer(
    mock_bedrock_agent_runtime,
    mock_bedrock_runtime,
    mock_event,
    mock_apply_guardrail_pass_response,
    mock_retrieve_response,
    mock_converse_no_answer_response,
    capsys,
    caplog,
):
    mock_event["body"] = '{"question": "fake question"}'

    mock_bedrock_runtime.apply_guardrail.return_value = mock_apply_guardrail_pass_response
    mock_bedrock_agent_runtime.retrieve.return_value = mock_retrieve_response
    mock_bedrock_runtime.converse.return_value = mock_converse_no_answer_response

    caplog.set_level(logging.WARNING)

    response = lambda_handler(mock_event, None)

    assert caplog.records[-1].message == "Unable to provide answer"
    assert isinstance(caplog.records[-1].exc_info[1], NoAnswerFromModelError)

    assert response.get("statusCode") == HTTPStatus.OK
    actual_response_json = json.loads(response.get("body"))
    assert "message" not in actual_response_json
    assert "error" in actual_response_json
    assert actual_response_json["error"]["code"] == "NoAnswerFromModelError"
    assert actual_response_json["error"]["message"] == "LLM unable to provide answer."
    assert (
        actual_response_json["error"]["model_response"] == "Sorry I'm not able to provide answer."
    )

    assert mock_bedrock_runtime.apply_guardrail.call_args_list[1] == call(
        guardrailIdentifier=GUARDRAIL_ID,
        guardrailVersion=GUARDRAIL_VERSION,
        source="OUTPUT",
        content=[
            {
                "text": {
                    "text": "Sorry I'm not able to provide answer.",
                    "qualifiers": ["guard_content"],
                }
            }
        ],
    )

    # When LLM output does not contain answer tags, the error message carries raw LLM output,
    # which should be checked by OUTPUT guardrails
    metrics = capture_metrics_output_multiple_emf_objects(capsys)
    assert metrics[0] == expected_guardrail_intervention_metrics("INPUT", 0)
    assert metrics[1] == expected_guardrail_intervention_metrics("OUTPUT", 0)
    assert_metrics_existed(
        metrics[2],
        {
            "NoAnswerFromModelError": 1,
            "InputTokens": 100,
            "OutputTokens": 20,
            "TotalTokens": 120,
        },
    )


def test_invalid_input_no_question(mock_event):
    """Test when the request body doesn't contain a question field."""
    mock_event["body"] = "{}"

    response = lambda_handler(mock_event, None)

    assert response.get("statusCode") == HTTPStatus.BAD_REQUEST
    actual_response_json = json.loads(response.get("body"))
    assert (
        actual_response_json["error"]["message"]
        == "Input validation failed. Errors: ['Field question: Field required']"
    )


def test_invalid_input_empty(mock_event):
    """Test when the question is empty."""
    mock_event["body"] = '{"question": ""}'

    response = lambda_handler(mock_event, None)

    assert response.get("statusCode") == HTTPStatus.BAD_REQUEST
    actual_response_json = json.loads(response.get("body"))
    assert (
        actual_response_json["error"]["message"]
        == "Input validation failed. Errors: ['Field question: String should have at least 1 character']"
    )


def test_invalid_input_too_long(mock_event):
    """Test when the question is too long."""
    mock_event["body"] = json.dumps({"question": "a" * 1001})

    response = lambda_handler(mock_event, None)

    assert response.get("statusCode") == HTTPStatus.BAD_REQUEST
    actual_response_json = json.loads(response.get("body"))
    assert (
        actual_response_json["error"]["message"]
        == "Input validation failed. Errors: ['Field question: String should have at most 1000 characters']"
    )


def test_retrieve_context(mock_retrieve_response):
    mock_bedrock_agent_runtime = MagicMock()
    mock_bedrock_agent_runtime.retrieve.return_value = mock_retrieve_response

    result = retrieve_context(mock_bedrock_agent_runtime, "test query")
    assert mock_bedrock_agent_runtime.retrieve.call_args == call(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": "test query"},
        retrievalConfiguration=ANY,
    )
    expected = [{"text": "Content Foo"}, {"text": "Content Bar"}]
    assert result == expected


def test_format_context():
    documents = ["Content Foo", "Content Bar"]
    expected = dedent(
        """\
        <context>
        <document>Content Foo</document><document>Content Bar</document>
        </context>
    """
    )
    result = format_context(documents)
    assert result == expected


def test_format_query():
    question = "fake question"
    expected = dedent(
        """\
        <question>
        fake question
        </question>
    """
    )
    result = format_query(question)
    assert result == expected


def test_converse_with_model(mock_converse_response):
    mock_bedrock_runtime = MagicMock()
    mock_bedrock_runtime.converse.return_value = mock_converse_response

    result = inference(mock_bedrock_runtime, "fake system prompt", "fake user prompt")
    assert mock_bedrock_runtime.converse.call_args == call(
        modelId=MODEL_ID,
        system=[{"text": "fake system prompt"}],
        messages=[{"role": "user", "content": [{"text": "fake user prompt"}]}],
        inferenceConfig=ANY,
    )
    assert result == "<answer>fake answer</answer>"


def test_parse_output():
    text = dedent(
        """\
    Some text
    <answer>
    Line 1
    Line 2
    </answer>
    more text"""
    )
    answer = parse_output(text)
    assert answer == "Line 1\nLine 2"


def test_parse_output_without_tags():
    text = "Sorry I'm not able to provide answer."
    answer = parse_output(text)
    assert not answer
