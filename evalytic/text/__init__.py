"""Text and RAG evaluation helpers."""

from .runner import evaluate_rag, evaluate_text, load_cases_from_dataset
from .types import (
    MetricEvalReport,
    MetricEvalResult,
    MetricResult,
    RAGTestCase,
    RetrievedChunk,
    TextEvalResult,
    TextTestCase,
)

__all__ = [
    "RetrievedChunk",
    "RAGTestCase",
    "TextTestCase",
    "MetricResult",
    "MetricEvalResult",
    "TextEvalResult",
    "MetricEvalReport",
    "evaluate_rag",
    "evaluate_text",
    "load_cases_from_dataset",
]
