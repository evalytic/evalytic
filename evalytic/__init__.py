"""Evalytic -- Evals for AI outputs.

Quick start (CLI):

    $ pip install evalytic
    $ export FAL_KEY=... GEMINI_API_KEY=...

    # Benchmark 3 models, get HTML report
    $ evaly bench -m flux-schnell -m flux-dev -m flux-pro \\
        -p "A cat on a windowsill" -o report.html --yes

    # Score a single image
    $ evaly eval --image output.png --prompt "A sunset over mountains"

Python API:

    import evalytic

    result = evalytic.eval_image(
        image="https://example.com/output.png",
        prompt="A cat wearing a top hat",
    )
    print(result.display_score)  # "3.8/5"
"""

__version__ = "0.4.1"

from .agent import AgentTestCase, AgentToolCall, evaluate_agent
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
from .text import (
    MetricEvalReport,
    MetricEvalResult,
    MetricResult,
    RAGTestCase,
    RetrievedChunk,
    TextEvalResult,
    TextTestCase,
    evaluate_rag,
    evaluate_text,
)
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
    "evaluate_rag",
    "evaluate_text",
    "evaluate_agent",
    # Classes
    "Eval",
    "Dataset",
    "init_dataset",
    "RetrievedChunk",
    "RAGTestCase",
    "TextTestCase",
    "MetricResult",
    "MetricEvalResult",
    "TextEvalResult",
    "MetricEvalReport",
    "AgentTestCase",
    "AgentToolCall",
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
