"""Task interface: a seeded template that builds a fresh world, a prompt, and a grader.

Every task also ships an ``oracle`` (the gold action sequence). It is used to:
  * compute the expected end state by replaying gold actions on a fresh world
    (tau-bench style state grading), and
  * drive the deterministic fake agents in tests and offline demos.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from agentarena.schemas import Score, ToolCall, Trajectory
from agentarena.tools import Environment


@dataclass
class OraclePlan:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    final_text: str = "Done."


class Task(ABC):
    template: ClassVar[str]
    domain: ClassVar[Literal["tool_use", "research"]]
    system_prompt: ClassVar[str]

    def __init__(self, seed: int):
        self.seed = seed
        # String seeds keep templates independent: changing one never shifts another.
        self.rng = random.Random(f"{self.template}:{seed}")
        self.prompt: str = ""
        self.setup()

    @property
    def task_id(self) -> str:
        return f"{self.template}#{self.seed}"

    @abstractmethod
    def setup(self) -> None:
        """Sample task parameters from ``self.rng`` and set ``self.prompt``."""

    @abstractmethod
    def build_env(self) -> Environment:
        """Return a fresh, deterministic environment (called once per sample)."""

    @abstractmethod
    def oracle(self, env: Environment) -> OraclePlan:
        """Gold actions that solve the task, given a fresh environment."""

    @abstractmethod
    def score(self, env: Environment, traj: Trajectory) -> Score:
        """Grade the final environment state and/or the submitted answer."""

    def replay_oracle(self) -> Environment:
        env = self.build_env()
        for i, (name, args) in enumerate(self.oracle(env).calls):
            res = env.execute(ToolCall(id=f"oracle-{i}", name=name, arguments=args))
            assert not res.is_error, f"{self.task_id}: oracle call {name} failed: {res.content}"
        return env
