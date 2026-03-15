"""Base provider ABC for LLM judge calls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseProvider(ABC):
    """Abstract base for LLM providers.

    All providers use sync httpx.Client. Consensus layer handles
    parallelism via ThreadPoolExecutor.
    """

    @abstractmethod
    def complete(
        self,
        user_prompt: str,
        system_prompt: str,
        images: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Send prompt (with optional images) and get parsed JSON response.

        Args:
            user_prompt: The evaluation prompt.
            system_prompt: System-level instructions.
            images: Optional list of (base64_data, mime_type) tuples.

        Returns:
            Parsed JSON dict from the LLM response.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release underlying HTTP client."""
        ...
