"""Tests for evalytic.exceptions -- hierarchy validation."""

from __future__ import annotations

import pytest

from evalytic.exceptions import (
    ConfigError,
    EvalyticError,
    GenerationError,
    JudgeError,
    ValidationError,
)


class TestHierarchy:
    def test_all_inherit_from_evalytic_error(self) -> None:
        for cls in (ConfigError, ValidationError, GenerationError, JudgeError):
            assert issubclass(cls, EvalyticError)

    def test_evalytic_error_is_exception(self) -> None:
        assert issubclass(EvalyticError, Exception)

    def test_validation_error_is_value_error(self) -> None:
        assert issubclass(ValidationError, ValueError)

    def test_unknown_model_error_is_validation_error(self) -> None:
        from evalytic.bench.registry import UnknownModelError

        assert issubclass(UnknownModelError, ValidationError)
        assert issubclass(UnknownModelError, ValueError)

    def test_config_error_not_value_error(self) -> None:
        assert not issubclass(ConfigError, ValueError)

    def test_catch_evalytic_error(self) -> None:
        with pytest.raises(EvalyticError):
            raise ValidationError("bad input")

    def test_catch_value_error_compat(self) -> None:
        with pytest.raises(ValueError):
            raise ValidationError("bad input")

    def test_catch_evalytic_catches_judge_error(self) -> None:
        with pytest.raises(EvalyticError):
            raise JudgeError("judge failed")

    def test_catch_evalytic_catches_generation_error(self) -> None:
        with pytest.raises(EvalyticError):
            raise GenerationError("gen failed")

    def test_catch_evalytic_catches_config_error(self) -> None:
        with pytest.raises(EvalyticError):
            raise ConfigError("config missing")
