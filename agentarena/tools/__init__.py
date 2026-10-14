"""Tool definitions and the per-sample environment that executes them."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import jsonschema

from agentarena.schemas import ToolCall, ToolResult


class ToolError(Exception):
    """Raised by a tool for an expected, model-visible failure (bad id, out of stock...)."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (object)
    fn: Callable[..., Any]

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False,
    }


@dataclass
class Environment:
    """Fresh, isolated world for one sample. Tasks grade its final ``state``."""

    tools: list[Tool]
    state: Any = None
    # Set by a terminal tool such as ``submit_answer``; ends the episode.
    submission: dict[str, Any] | None = None
    _by_name: dict[str, Tool] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_name = {t.name: t for t in self.tools}

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self.tools]

    def execute(self, call: ToolCall) -> ToolResult:
        def err(msg: str, kind: str) -> ToolResult:
            return ToolResult(
                call_id=call.id, name=call.name, content=f"ERROR: {msg}", is_error=True, error_kind=kind
            )

        tool = self._by_name.get(call.name)
        if tool is None:
            return err(f"unknown tool '{call.name}'", "unknown_tool")
        if call.raw_arguments is not None:
            return err("arguments were not valid JSON", "bad_json")
        try:
            jsonschema.validate(call.arguments, tool.parameters)
        except jsonschema.ValidationError as e:
            return err(f"invalid arguments: {e.message}", "schema")
        try:
            out = tool.fn(**call.arguments)
        except ToolError as e:
            return err(str(e), "domain")
        content = out if isinstance(out, str) else json.dumps(out, sort_keys=True)
        return ToolResult(call_id=call.id, name=call.name, content=content)
