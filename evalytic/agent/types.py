"""Types for agent evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentToolCall:
    name: str
    arguments: dict[str, Any] | None = None
    output: str | None = None
    status: str = "success"


@dataclass
class AgentTestCase:
    input: str
    final_output: str
    expected_output: str | None = None
    tool_calls: list[AgentToolCall] | None = None
    metadata: dict[str, Any] | None = None
