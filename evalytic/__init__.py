"""Evalytic -- Visual Generation Quality Evaluation SDK.

Quick start:

    import evalytic

    evalytic.init(api_key="ek-...")

    # Single image evaluation
    result = evalytic.eval_image(
        image="https://example.com/output.png",
        prompt="A cat wearing a top hat",
    )
    print(result.display_score)  # "3.8/5"

    # Full experiment with a dataset
    ds = evalytic.init_dataset("my-project", "hero-prompts")
    ds.insert({"prompt": "A cat in a top hat"})
    ds.insert({"prompt": "Sunset over Tokyo"})

    experiment = evalytic.Eval(
        project="my-project",
        data=ds,
        scores=["visual_quality", "prompt_adherence"],
        model="dall-e-3",
    )
    print(experiment.summary)
"""

__version__ = "0.2.1"

from .client import _get_client, init
from .compare import compare
from .dataset import Dataset, init_dataset
from .eval import eval_image
from .exceptions import (
    ConfigError,
    EvalyticError,
    GenerationError,
    JudgeError,
    ValidationError,
)
from .experiment import Eval
from .types import CompareReport, DimensionScore, EvalResult, ExperimentResult
from .bench.registry import register_model
from .bench.runner import run_bench as bench

__all__ = [
    "__version__",
    # Client
    "init",
    "_get_client",
    # Core functions
    "eval_image",
    "compare",
    "bench",
    "register_model",
    # Classes
    "Eval",
    "Dataset",
    "init_dataset",
    # Types
    "DimensionScore",
    "EvalResult",
    "ExperimentResult",
    "CompareReport",
    # Exceptions
    "EvalyticError",
    "ConfigError",
    "ValidationError",
    "GenerationError",
    "JudgeError",
]
