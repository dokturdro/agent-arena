"""Task registry."""

from __future__ import annotations

from agentarena.tasks.base import OraclePlan, Task
from agentarena.tasks.research.tasks import TEMPLATES as RESEARCH
from agentarena.tasks.tool_use.tasks import TEMPLATES as TOOL_USE

REGISTRY: dict[str, type[Task]] = {t.template: t for t in [*TOOL_USE, *RESEARCH]}


def build_tasks(templates: list[str] | str, seeds: list[int]) -> list[Task]:
    if templates == "all":
        names = list(REGISTRY)
    elif templates in ("tool_use", "research"):
        names = [n for n, t in REGISTRY.items() if t.domain == templates]
    else:
        unknown = set(templates) - set(REGISTRY)
        if unknown:
            raise ValueError(f"unknown templates: {sorted(unknown)}")
        names = list(templates)
    return [REGISTRY[n](seed) for n in names for seed in seeds]


__all__ = ["REGISTRY", "OraclePlan", "Task", "build_tasks"]
