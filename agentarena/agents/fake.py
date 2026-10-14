"""Deterministic offline models, driven by each task's oracle.

They exist so the whole pipeline (runner, scoring, stats, rating, report) can be
tested and demoed with no API keys and no cost:

* ``oracle``  - replays the gold plan (should score 100%)
* ``noisy``   - follows the plan but, with probability ``p_fail`` per step, quits
                early, corrupts an argument, or makes an invalid call and retries
* ``lazy``    - answers immediately without using tools
"""

from __future__ import annotations

import json
import random
from typing import Any

from agentarena.agents.base import Model, ModelConfig
from agentarena.schemas import Completion, Message, ToolCall, Usage


class FakeModel(Model):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.policy = cfg.model  # oracle | noisy | lazy
        self.p_fail = float(cfg.params.get("p_fail", 0.15))
        self._calls: list[tuple[str, dict[str, Any]]] = []
        self._final = ""
        self._rng = random.Random(0)

    def for_sample(self, task, env, salt: str) -> Model:
        m = FakeModel(self.cfg)
        plan = task.oracle(env)
        m._calls, m._final = list(plan.calls), plan.final_text
        m._rng = random.Random(f"{self.name}:{salt}")
        return m

    async def complete(self, messages: list[Message], tools: list[dict[str, Any]], *, max_tokens: int) -> Completion:
        turn = sum(m.role == "assistant" for m in messages)
        usage = Usage(input_tokens=300 + 250 * turn, output_tokens=80)
        if self.policy == "lazy":
            return Completion(text="I think that's done.", usage=usage)
        if turn >= len(self._calls):
            return Completion(text=self._final, usage=usage)

        name, args = self._calls[turn]
        args = json.loads(json.dumps(args))
        if self.policy == "noisy":
            r = self._rng.random()
            if r < self.p_fail * 0.2:
                return Completion(text="I believe the task is complete.", usage=usage)  # quits early
            if r < self.p_fail * 0.6:
                args = _corrupt(args, self._rng)
            elif r < self.p_fail:
                # Invalid call; re-queue the real one so the agent "recovers" next turn.
                self._calls.insert(turn, (name, args))
                return Completion(tool_calls=[ToolCall(id=f"c{turn}", name=name, arguments={"bogus": 1})], usage=usage)
        return Completion(tool_calls=[ToolCall(id=f"c{turn}", name=name, arguments=args)], usage=usage)


def _corrupt(args: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    for k, v in args.items():
        if isinstance(v, int) and not isinstance(v, bool):
            args[k] = v + 1
            return args
        if isinstance(v, str) and k in {"sku", "order_id", "event_id", "answer", "start", "date"}:
            args[k] = v[:-1] + ("7" if not v.endswith("7") else "3")
            return args
    return args
