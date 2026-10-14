"""Model adapter interface and configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from agentarena.schemas import Completion, Message

if TYPE_CHECKING:
    from agentarena.tasks import Task
    from agentarena.tools import Environment


class Price(BaseModel):
    input: float = 0.0  # USD per 1M tokens
    output: float = 0.0


class ModelConfig(BaseModel):
    name: str  # display name / leaderboard key
    provider: str  # anthropic | openai_compat | fake
    model: str  # provider model id
    family: str | None = None  # for judge != contestant family; defaults to provider
    base_url: str | None = None
    api_key_env: str | None = None
    price: Price = Field(default_factory=Price)
    concurrency: int = 4
    params: dict[str, Any] = Field(default_factory=dict)  # extra request params, passed through

    @property
    def model_family(self) -> str:
        return self.family or self.provider

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.price.input + output_tokens * self.price.output) / 1e6


class ProviderError(Exception):
    """A non-retryable provider failure for this sample (bad request, auth...)."""


class Model(ABC):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    @property
    def name(self) -> str:
        return self.cfg.name

    def for_sample(self, task: Task, env: Environment, salt: str) -> Model:
        """Per-sample instance. API models are stateless and return self; fakes bind the oracle."""
        return self

    @abstractmethod
    async def complete(self, messages: list[Message], tools: list[dict[str, Any]], *, max_tokens: int) -> Completion:
        """Run one model turn on a neutral message list and neutral tool specs."""
