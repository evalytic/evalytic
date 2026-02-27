"""Evalytic exception hierarchy.

All Evalytic-specific errors inherit from :class:`EvalyticError` so callers
can ``except EvalyticError`` to catch everything.

:class:`ValidationError` also inherits from :class:`ValueError` for
backwards compatibility with existing ``except ValueError`` handlers.
"""


class EvalyticError(Exception):
    """Base exception for all Evalytic errors."""


class ConfigError(EvalyticError):
    """Raised for configuration file errors (missing file, parse error)."""


class ValidationError(EvalyticError, ValueError):
    """Raised for invalid arguments (unknown dimensions, bad model names, etc.)."""


class GenerationError(EvalyticError):
    """Raised for image generation failures (API errors, missing dependencies)."""


class JudgeError(EvalyticError):
    """Raised when the VLM judge fails after retries."""
