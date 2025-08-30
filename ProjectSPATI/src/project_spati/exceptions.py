import json
from abc import ABC, abstractmethod


class UnableToProvideAnswerError(Exception, ABC):
    """Base exception class when service cannot return answer."""

    @abstractmethod
    def error_message(self) -> str:
        pass  # pragma: no cover


class NoContextFoundError(UnableToProvideAnswerError):
    """Raised when no context is found for the user query."""

    def __init__(self):
        super().__init__("No context found for user query")

    def error_message(self) -> str:
        return "No context found for user query."


class NoAnswerFromModelError(UnableToProvideAnswerError):
    """Raised when the LLM does not provide an answer to the question."""

    def __init__(self, llm_output: str):
        self.llm_output = llm_output
        super().__init__("Unable to extract answer from LLM output")

    def error_message(self) -> str:
        return "LLM unable to provide answer."


class GuardrailInterventionError(UnableToProvideAnswerError):
    """Raised when the Guardrail intervenes."""

    def __init__(self, source_type: str, assessments: list):
        self.source_type = source_type
        self.assessments = assessments
        super().__init__()

    def error_message(self) -> str:
        # Only keep the assessment. No need to keep the invocation metrics.
        trimmed_assessment = [
            {k: v for (k, v) in a.items() if k != "invocationMetrics"} for a in self.assessments
        ]
        return (
            f"Guardrail intervened on {self.source_type}. "
            f"Assessment: {json.dumps(trimmed_assessment)}"
        )
