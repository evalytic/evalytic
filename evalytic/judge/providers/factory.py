"""Provider configuration, parsing, and factory."""

from __future__ import annotations

import os
from typing import Any

from ...exceptions import ValidationError
from .anthropic import AnthropicProvider
from .base import BaseProvider
from .gemini import GeminiProvider
from .openai_compat import OpenAICompatProvider

# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "env_key": "GEMINI_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "env_key": None,
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "env_key": None,
    },
    "local": {
        "base_url": "http://localhost:8090/v1",
        "env_key": None,
    },
    "fal": {
        "base_url": "https://fal.run/openrouter/router/openai/v1",
        "env_key": "FAL_KEY",
    },
}

# Providers that use OpenAI-compatible API format
_OPENAI_COMPAT_PROVIDERS = {"openai", "ollama", "lmstudio", "local", "fal"}

# Short aliases → actual Gemini API model names.
_GEMINI_MODEL_ALIASES: dict[str, str] = {
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
}

# fal.ai OpenRouter model name mapping.
_FAL_MODEL_MAP: dict[str, str] = {
    # Gemini
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-3-flash": "google/gemini-3-flash-preview",
    # OpenAI
    "gpt-5.2": "openai/gpt-5.2",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    # Anthropic
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
    "claude-haiku-4-5": "anthropic/claude-haiku-4-5",
    "claude-opus-4-6": "anthropic/claude-opus-4-6",
}

# Recommended judge models by provider (for docs/CLI help)
RECOMMENDED_JUDGES: dict[str, list[str]] = {
    "gemini": [
        "gemini-3-flash",
        "gemini-3.1-pro",
        "gemini-2.5-flash",
    ],
    "openai": [
        "gpt-5.2",
        "o4-mini",
    ],
    "anthropic": [
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5",
    ],
    "fal": [
        "fal/gemini-2.5-flash",
        "fal/gpt-5.2",
        "fal/claude-sonnet-4-6",
    ],
    "self-hosted": [
        "ollama/qwen3-vl",
        "ollama/internvl3",
        "ollama/glm-4.1v-9b-thinking",
    ],
}

# Prefixes that map to OpenAI provider
_OPENAI_PREFIXES = ("gpt", "o1", "o3", "o4", "chatgpt")


def _parse_judge_string(judge: str) -> tuple[str, str]:
    """Parse a judge string into (provider, model).

    Examples:
        "gemini-3-flash"            -> ("gemini", "gemini-3-flash")
        "gpt-5.2"                   -> ("openai", "gpt-5.2")
        "claude-sonnet-4-6"         -> ("anthropic", "claude-sonnet-4-6")
        "fal/gemini-2.5-flash"      -> ("fal", "gemini-2.5-flash")
        "ollama/qwen3-vl"           -> ("ollama", "qwen3-vl")
    """
    if "/" in judge:
        provider, model = judge.split("/", 1)
        if provider not in _PROVIDER_DEFAULTS:
            raise ValidationError(
                f"Unknown judge provider: {provider!r}. "
                f"Valid: {sorted(_PROVIDER_DEFAULTS)}"
            )
        return provider, model

    # Bare model name -- infer provider from prefix
    if judge.startswith("gemini"):
        return "gemini", judge
    for prefix in _OPENAI_PREFIXES:
        if judge.startswith(prefix):
            return "openai", judge
    if judge.startswith("claude"):
        return "anthropic", judge

    # Default to gemini for backward compat
    return "gemini", judge


def create_provider(
    judge: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[BaseProvider, str, str]:
    """Create a provider instance from a judge string.

    Returns:
        (provider_instance, provider_name, resolved_model_name)
    """
    provider_name, model = _parse_judge_string(judge)

    # Resolve short aliases
    if provider_name == "gemini":
        model = _GEMINI_MODEL_ALIASES.get(model, model)
    elif provider_name == "fal":
        model = _FAL_MODEL_MAP.get(model, model)

    defaults = _PROVIDER_DEFAULTS[provider_name]
    resolved_base_url = base_url or defaults["base_url"]

    env_key = defaults["env_key"]
    resolved_api_key = api_key or (os.environ.get(env_key, "") if env_key else "")

    if provider_name == "gemini":
        provider = GeminiProvider(model, resolved_api_key, resolved_base_url)
    elif provider_name in _OPENAI_COMPAT_PROVIDERS:
        provider = OpenAICompatProvider(model, resolved_api_key, resolved_base_url, provider_name)
    elif provider_name == "anthropic":
        provider = AnthropicProvider(model, resolved_api_key, resolved_base_url)
    else:
        raise ValidationError(f"Unknown provider: {provider_name!r}")

    return provider, provider_name, model
