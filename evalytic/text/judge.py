"""Shared LLM-as-judge wrapper for text-like eval domains."""

from __future__ import annotations

from typing import Any

from ..judge.common import call_with_retries
from ..judge.providers import create_provider

SYSTEM_PROMPT = (
    "You are an expert evaluator for AI outputs.\n"
    "You analyze text, RAG answers, and agent traces using the provided rubric.\n"
    "Always return valid JSON that matches the requested structure.\n"
    "Be precise, evidence-based, and avoid speculation.\n"
)


class TextJudge:
    """Structured text evaluation wrapper around the shared provider stack."""

    def __init__(
        self,
        judge: str = "gemini-2.5-flash",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._provider, self.provider, self.model = create_provider(
            judge,
            api_key=api_key,
            base_url=base_url,
        )
        self.judge_string = judge

    @property
    def last_cost(self) -> float:
        return float(getattr(self._provider, "last_cost", 0.0) or 0.0)

    def complete_json(
        self,
        user_prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return call_with_retries(
            self._provider.complete,
            user_prompt,
            system_prompt,
            images=None,
            response_schema=response_schema,
        )

    def close(self) -> None:
        self._provider.close()

    def __enter__(self) -> TextJudge:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
