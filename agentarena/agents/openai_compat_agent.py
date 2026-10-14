"""OpenAI-compatible Chat Completions adapter: Z.ai (GLM), Ollama, OpenAI, vLLM, ..."""

from __future__ import annotations

import json
import os
from typing import Any

import openai

from agentarena.agents.base import Model, ModelConfig, ProviderError
from agentarena.schemas import Completion, Message, ToolCall, Usage


class OpenAICompatModel(Model):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else "not-needed"
        if cfg.api_key_env and not key:
            raise ProviderError(f"{cfg.name}: environment variable {cfg.api_key_env} is not set")
        self.client = openai.AsyncOpenAI(api_key=key, base_url=cfg.base_url, max_retries=6)

    @staticmethod
    def to_wire(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role in ("system", "user"):
                out.append({"role": m.role, "content": m.content})
            elif m.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {"id": c.id, "type": "function", "function": {
                            "name": c.name, "arguments": c.raw_arguments or json.dumps(c.arguments)}}
                        for c in m.tool_calls
                    ]
                out.append(msg)
            elif m.role == "tool":
                out += [{"role": "tool", "tool_call_id": r.call_id, "content": r.content} for r in m.tool_results]
        return out

    async def complete(self, messages: list[Message], tools: list[dict[str, Any]], *, max_tokens: int) -> Completion:
        extra: dict[str, Any] = dict(self.cfg.params)
        if tools:
            extra["tools"] = [{"type": "function", "function": t} for t in tools]
        try:
            resp = await self.client.chat.completions.create(
                model=self.cfg.model, messages=self.to_wire(messages), max_tokens=max_tokens, **extra
            )
        except (openai.BadRequestError, openai.AuthenticationError,
                openai.PermissionDeniedError, openai.NotFoundError) as e:
            raise ProviderError(f"{type(e).__name__}: {e}") from e

        choice = resp.choices[0]
        calls = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
                raw = None if isinstance(args, dict) else tc.function.arguments
            except json.JSONDecodeError:
                args, raw = {}, tc.function.arguments
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args if isinstance(args, dict) else {},
                                  raw_arguments=raw))
        u = resp.usage
        return Completion(
            text=choice.message.content or "",
            tool_calls=calls,
            usage=Usage(input_tokens=u.prompt_tokens if u else 0, output_tokens=u.completion_tokens if u else 0),
            stop_reason=choice.finish_reason or "stop",
        )
