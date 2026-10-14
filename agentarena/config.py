"""YAML config loading: models.yaml (contestants) and suites/*.yaml (what to run)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from agentarena.agents.base import ModelConfig
from agentarena.agents.loop import Budget

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


class Suite(BaseModel):
    name: str
    templates: list[str] | str = "all"  # "all" | "tool_use" | "research" | [names]
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3])
    repeats: int = 3
    budget: Budget = Field(default_factory=Budget)


def load_models(path: Path | None = None) -> dict[str, ModelConfig]:
    data = yaml.safe_load((path or CONFIG_DIR / "models.yaml").read_text(encoding="utf-8"))
    return {m["name"]: ModelConfig(**m) for m in data["models"]}


def load_suite(name_or_path: str) -> Suite:
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIG_DIR / "suites" / f"{name_or_path}.yaml"
    data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))
    return Suite(**data)
