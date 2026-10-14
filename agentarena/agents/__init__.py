"""Model adapters."""

from __future__ import annotations

from agentarena.agents.base import Model, ModelConfig, ProviderError


def make_model(cfg: ModelConfig) -> Model:
    if cfg.provider == "anthropic":
        from agentarena.agents.anthropic_agent import AnthropicModel

        return AnthropicModel(cfg)
    if cfg.provider == "openai_compat":
        from agentarena.agents.openai_compat_agent import OpenAICompatModel

        return OpenAICompatModel(cfg)
    if cfg.provider == "fake":
        from agentarena.agents.fake import FakeModel

        return FakeModel(cfg)
    raise ValueError(f"unknown provider {cfg.provider!r}")


__all__ = ["Model", "ModelConfig", "ProviderError", "make_model"]
