"""HTTP client for the Evalytic API.

Usage:
    import evalytic
    evalytic.init(api_key="ek-...")  # or set EVALYTIC_API_KEY env var
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

_DEFAULT_API_URL = "https://api.evalytic.dev"
_DEFAULT_TIMEOUT = 30.0


@dataclass
class _Config:
    api_key: str | None = None
    api_url: str | None = None


_config = _Config()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def init(
    api_key: str | None = None,
    api_url: str | None = None,
) -> None:
    """Initialise the Evalytic SDK with credentials.

    Parameters
    ----------
    api_key:
        API key for authentication.  Falls back to the ``EVALYTIC_API_KEY``
        environment variable when *None*.
    api_url:
        Base URL of the Evalytic API.  Falls back to ``EVALYTIC_API_URL``
        then to the default production URL.
    """
    _config.api_key = api_key
    _config.api_url = api_url


def _get_client() -> httpx.Client:
    """Return a configured ``httpx.Client`` pointing at the Evalytic API.

    The client reads configuration in the following order of precedence:
    1. Values passed to ``init()``.
    2. Environment variables ``EVALYTIC_API_KEY`` / ``EVALYTIC_API_URL``.
    3. Built-in defaults (production URL, no key).
    """
    api_key = _config.api_key or os.environ.get("EVALYTIC_API_KEY", "")
    api_url = _config.api_url or os.environ.get("EVALYTIC_API_URL", _DEFAULT_API_URL)

    if not api_key:
        raise RuntimeError(
            "Evalytic API key is not set. "
            "Call evalytic.init(api_key='ek-...') or set the EVALYTIC_API_KEY "
            "environment variable."
        )

    headers: dict[str, str] = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
        "User-Agent": "evalytic-python/0.2.1",
    }

    return httpx.Client(
        base_url=api_url,
        headers=headers,
        timeout=_DEFAULT_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Low-level request helpers (used by eval, experiment, dataset modules)
# ---------------------------------------------------------------------------


def _post(path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
    """Send a POST request and return the parsed JSON response."""
    with _get_client() as client:
        response = client.post(path, json=json)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Send a GET request and return the parsed JSON response."""
    with _get_client() as client:
        response = client.get(path, params=params)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
