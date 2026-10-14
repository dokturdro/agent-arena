"""Anthropic Messages API adapter (native tool use)."""

from __future__ import annotations

import os
from typing import Any

import anthropic

from agentarena.agents.base import Model, ModelConfig, ProviderError
from agentarena.schemas import Completion, Message, ToolCall, Usage


class AnthropicModel(Model):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
        # Keys not scoped to a workspace must name one on every request.
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        headers = {"anthropic-workspace-id": workspace} if workspace else None
        # The SDK retries 408/409/429/5xx and connection errors with exponential backoff.
        self.client = anthropic.AsyncAnthropic(api_key=key, base_url=cfg.base_url, max_retries=6,
                                               default_headers=headers)

    @staticmethod
    def to_wire(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                if m.provider_raw is not None:  # replay verbatim (thinking blocks, signatures)
                    out.append({"role": "assistant", "content": m.provider_raw})
                    continue
                blocks: list[dict[str, Any]] = [{"type": "text", "text": m.content}] if m.content else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls]
                out.append({"role": "assistant", "content": blocks})
            elif m.role == "tool":
                # All results of one turn go back in a single user message.
                out.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content, "is_error": r.is_error}
                    for r in m.tool_results
                ]})
        return system, out

    async def complete(self, messages: list[Message], tools: list[dict[str, Any]], *, max_tokens: int) -> Completion:
        system, wire = self.to_wire(messages)
        extra: dict[str, Any] = dict(self.cfg.params)
        if system:
            extra["system"] = system
        if tools:
            extra["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                              for t in tools]
        try:
            resp = await self.client.messages.create(
                model=self.cfg.model, max_tokens=max_tokens, messages=wire, **extra
            )
        except (anthropic.BadRequestError, anthropic.AuthenticationError,
                anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
            raise ProviderError(f"{type(e).__name__}: {e.message}") from e

        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, arguments=dict(b.input)) for b in resp.content if b.type == "tool_use"]
        u = resp.usage
        input_tokens = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
        return Completion(
            text=text,
            tool_calls=calls,
            usage=Usage(input_tokens=input_tokens, output_tokens=u.output_tokens),
            stop_reason=resp.stop_reason or "end_turn",
            provider_raw=[b.model_dump(mode="json", exclude_none=True) for b in resp.content],
        )
