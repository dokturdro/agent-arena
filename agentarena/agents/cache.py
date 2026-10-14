"""Content-addressed disk cache for model turns.

The key covers the model id, request params, the full message history, the tool
specs and a per-sample salt (task id + repeat). Re-running or resuming a run is
then free, while repeats remain independent samples.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agentarena.agents.base import Model
from agentarena.schemas import Completion, Message


class DiskCache:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Completion | None:
        p = self._path(key)
        if p.exists():
            return Completion.model_validate_json(p.read_text(encoding="utf-8"))
        return None

    def put(self, key: str, comp: Completion) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(comp.model_dump_json(), encoding="utf-8")
        tmp.replace(p)


class CachedModel(Model):
    def __init__(self, inner: Model, cache: DiskCache, salt: str):
        super().__init__(inner.cfg)
        self.inner, self.cache, self.salt = inner, cache, salt

    async def complete(self, messages: list[Message], tools: list[dict[str, Any]], *, max_tokens: int) -> Completion:
        payload = {
            "model": self.cfg.model,
            "provider": self.cfg.provider,
            "params": self.cfg.params,
            "max_tokens": max_tokens,
            "messages": [m.model_dump(mode="json") for m in messages],
            "tools": tools,
            "salt": self.salt,
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        hit = self.cache.get(key)
        if hit is not None:
            return hit.model_copy(update={"cached": True})
        comp = await self.inner.complete(messages, tools, max_tokens=max_tokens)
        self.cache.put(key, comp)
        return comp
