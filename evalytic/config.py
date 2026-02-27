"""Configuration loader for evalytic.toml files.

Search order: --config flag > ./evalytic.toml > ~/.evalytic/config.toml > defaults.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .exceptions import ConfigError


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file, using stdlib tomllib (3.11+) or tomli fallback."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return {}

    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load configuration from a TOML file.

    Parameters
    ----------
    path:
        Explicit path to config file (from ``--config`` flag).
        If *None*, searches ``./evalytic.toml`` then ``~/.evalytic/config.toml``.

    Returns
    -------
    dict
        Parsed config dict.  Empty dict if no file found.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Config file not found: {path}")
        return _load_toml(p)

    # Search order
    candidates = [
        Path("evalytic.toml"),
        Path.home() / ".evalytic" / "config.toml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return _load_toml(candidate)

    return {}


def apply_keys(config: dict[str, Any]) -> None:
    """Set API keys from config [keys] section as environment variables.

    Only sets a key if it is not already present in the environment,
    so env vars / .env always take precedence.
    """
    keys = config.get("keys", {})
    key_map = {
        "fal": "FAL_KEY",
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    for toml_key, env_var in key_map.items():
        value = keys.get(toml_key)
        if value and not os.environ.get(env_var):
            os.environ[env_var] = value
