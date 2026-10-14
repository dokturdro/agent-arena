"""Provider-neutral data model shared by agents, tasks, scorers and the report."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Raw argument string when the provider sent unparseable JSON.
    raw_arguments: str | None = None


class ToolResult(BaseModel):
    call_id: str
    name: str
    content: str
    is_error: bool = False
    # Protocol errors (the model misused the interface) vs domain errors (the world said no).
    error_kind: Literal["unknown_tool", "bad_json", "schema", "domain"] | None = None


class Message(BaseModel):
    """Neutral chat message. Adapters translate to/from provider formats."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)  # role == "tool"
    # Provider-native assistant content, replayed verbatim so history stays append-only
    # (e.g. Anthropic thinking blocks must be echoed back unchanged).
    provider_raw: Any = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class Completion(BaseModel):
    """One model turn."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    stop_reason: str = "end_turn"
    cached: bool = False
    provider_raw: Any = None


class Step(BaseModel):
    index: int
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0
    cached: bool = False


StopReason = Literal["final", "max_steps", "max_tokens", "timeout", "error", "refusal"]


class Trajectory(BaseModel):
    steps: list[Step] = Field(default_factory=list)
    final_answer: str = ""
    stop_reason: StopReason = "final"
    error: str | None = None
    usage: Usage = Field(default_factory=Usage)
    wall_time_s: float = 0.0

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [c for s in self.steps for c in s.tool_calls]

    @property
    def tool_results(self) -> list[ToolResult]:
        return [r for s in self.steps for r in s.tool_results]


class Score(BaseModel):
    success: bool
    score: float  # 0..1, partial credit
    details: dict[str, Any] = Field(default_factory=dict)


class SampleRecord(BaseModel):
    """One (model, task instance, repeat) execution: the unit stored in samples.jsonl."""

    run_id: str
    sample_id: str
    model: str
    task_id: str  # template + seed, identical across models
    template: str
    domain: Literal["tool_use", "research"]
    seed: int
    repeat: int
    prompt: str
    trajectory: Trajectory
    outcome: Score
    metrics: dict[str, float | None] = Field(default_factory=dict)  # None = undefined (e.g. no errors to recover)
    cost_usd: float = 0.0
