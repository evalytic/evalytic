"""Multi-model comparison.

Runs the same dataset through multiple models and produces a ranked
``CompareReport``.

Example:
    import evalytic
    from evalytic.scorers import VISUAL_QUALITY, PROMPT_ADHERENCE

    evalytic.init(api_key="ek-...")

    ds = evalytic.init_dataset("my-project", "hero-prompts")
    ds.insert({"prompt": "A cat in a top hat"})
    ds.insert({"prompt": "Sunset over Tokyo"})

    report = evalytic.compare(
        project="my-project",
        data=ds,
        models=["dall-e-3", "midjourney-v6", "stable-diffusion-xl"],
        scores=[VISUAL_QUALITY, PROMPT_ADHERENCE],
    )
    print(report.winner)    # "midjourney-v6"
    print(report.ranking)   # [("midjourney-v6", 4.5), ("dall-e-3", 4.2), ...]
"""

from __future__ import annotations

from .dataset import Dataset
from .experiment import Eval
from .types import CompareReport, ExperimentResult


def compare(
    project: str,
    data: Dataset | str,
    models: list[str],
    scores: list[str],
    timeout: int = 600,
) -> CompareReport:
    """Run the same evaluation across multiple models and determine a winner.

    Parameters
    ----------
    project:
        The project identifier.
    data:
        A ``Dataset`` instance or a ``dataset_id`` string.  If a ``Dataset``
        is provided it will be pushed once and reused for every model.
    models:
        List of model identifiers to compare.
    scores:
        Evaluation dimensions to include.
    timeout:
        Per-model timeout in seconds.

    Returns
    -------
    CompareReport
        Contains per-model ``ExperimentResult`` objects and the overall
        ``winner`` (model with the highest score).
    """
    if not models:
        raise ValueError("At least one model must be provided for comparison.")

    # Ensure the dataset is pushed exactly once.
    if isinstance(data, Dataset) and not data.is_pushed:
        data.push()

    dataset_ref: Dataset | str = data

    results: dict[str, ExperimentResult] = {}

    for model in models:
        experiment = Eval(
            project=project,
            data=dataset_ref,
            scores=scores,
            model=model,
            experiment_name=f"compare-{model}",
            wait=True,
            timeout=timeout,
        )
        results[model] = experiment.result

    # Determine the winner by highest overall score.
    winner = ""
    best_score = -1.0
    for model, result in results.items():
        if result.score > best_score:
            best_score = result.score
            winner = model

    return CompareReport(
        models=list(models),
        results=results,
        winner=winner,
    )
