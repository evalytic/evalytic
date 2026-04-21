"""Metric registry for text, RAG, and related eval domains."""

from .answer_relevancy import AnswerRelevancyMetric
from .base import BaseTextMetric
from .context_precision import ContextPrecisionMetric
from .context_recall import ContextRecallMetric
from .factual_correctness import FactualCorrectnessMetric
from .faithfulness import FaithfulnessMetric
from .g_eval import GEvalMetric
from .semantic_similarity import SemanticSimilarityMetric
from .statistical import (
    BleuMetric,
    ExactMatchMetric,
    LevenshteinMetric,
    RougeMetric,
    StringPresenceMetric,
)

METRIC_REGISTRY = {
    "faithfulness": FaithfulnessMetric,
    "answer_relevancy": AnswerRelevancyMetric,
    "context_precision": ContextPrecisionMetric,
    "context_recall": ContextRecallMetric,
    "factual_correctness": FactualCorrectnessMetric,
    "semantic_similarity": SemanticSimilarityMetric,
    "g_eval": GEvalMetric,
    "bleu": BleuMetric,
    "rouge": RougeMetric,
    "exact_match": ExactMatchMetric,
    "levenshtein": LevenshteinMetric,
    "string_presence": StringPresenceMetric,
}

__all__ = [
    "BaseTextMetric",
    "AnswerRelevancyMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "FactualCorrectnessMetric",
    "FaithfulnessMetric",
    "GEvalMetric",
    "SemanticSimilarityMetric",
    "BleuMetric",
    "RougeMetric",
    "ExactMatchMetric",
    "LevenshteinMetric",
    "StringPresenceMetric",
    "METRIC_REGISTRY",
]
