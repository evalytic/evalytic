"""Experiment runner -- the primary way to evaluate a full dataset.

Example:
    import evalytic
    from evalytic.scorers import VISUAL_QUALITY, PROMPT_ADHERENCE

    evalytic.init(api_key="ek-...")

    ds = evalytic.init_dataset("my-project", "prompts")
    ds.insert({"prompt": "A cat in a top hat"})
    ds.insert({"prompt": "Sunset over Tokyo"})

    experiment = evalytic.Eval(
        project="my-project",
        data=ds,
        scores=[VISUAL_QUALITY, PROMPT_ADHERENCE],
        model="dall-e-3",
        experiment_name="dalle3-prompt-test",
    )
    print(experiment.summary)   # {"visual_quality": 4.2, "prompt_adherence": 3.9}
    print(experiment.status)    # "completed"
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from .client import _get, _post
from .dataset import Dataset
from .eval import _parse_eval_result
from .types import EvalResult, ExperimentResult


def _parse_experiment(data: dict[str, Any]) -> ExperimentResult:
    """Convert a raw API response into an ``ExperimentResult``."""
    eval_results: list[EvalResult] = []
    for r in data.get("results", []):
        eval_results.append(_parse_eval_result(r))

    return ExperimentResult(
        experiment_id=data.get("experiment_id", ""),
        name=data.get("name", ""),
        status=data.get("status", "processing"),
        score=float(data.get("score", 0.0)),
        total_items=int(data.get("total_items", 0)),
        completed_items=int(data.get("completed_items", 0)),
        scores=eval_results,
        summary=data.get("summary", {}),
    )


def _poll_experiment(
    experiment_id: str,
    timeout: int,
    interval: float = 3.0,
) -> ExperimentResult:
    """Poll until the experiment reaches a terminal state or we time out."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        data = _get(f"/v1/experiments/{experiment_id}")
        result = _parse_experiment(data)

        if result.is_complete:
            return result

        time.sleep(interval)

    # Timed out.
    return ExperimentResult(
        experiment_id=experiment_id,
        name="",
        status="timeout",
        score=0.0,
        total_items=0,
        completed_items=0,
    )


class Eval:
    """Run an evaluation experiment against a dataset.

    Instantiating this class kicks off the experiment immediately.  When
    ``wait=True`` (the default) the constructor blocks until the experiment
    completes or the timeout elapses.

    Parameters
    ----------
    project:
        The project identifier.
    data:
        A ``Dataset`` instance (will be pushed automatically if it hasn't
        been yet) **or** a ``dataset_id`` string for an already-uploaded
        dataset.
    scores:
        List of dimension names to evaluate (e.g.
        ``["visual_quality", "prompt_adherence"]``).
    model:
        The model identifier used for generation.
    experiment_name:
        Optional human-readable name.  Auto-generated when *None*.
    wait:
        Block until the experiment finishes.
    timeout:
        Maximum seconds to wait.
    """

    def __init__(
        self,
        project: str,
        data: Dataset | str,
        scores: list[str],
        model: str = "unknown",
        experiment_name: str | None = None,
        wait: bool = True,
        timeout: int = 300,
    ) -> None:
        self._project = project
        self._model = model
        self._scores = scores
        self._timeout = timeout
        self._experiment_name = experiment_name or f"eval-{uuid.uuid4().hex[:8]}"

        # Resolve the dataset_id ------------------------------------------
        if isinstance(data, Dataset):
            if not data.is_pushed:
                data.push()
            self._dataset_id: str = data.dataset_id  # type: ignore[assignment]
        else:
            self._dataset_id = data

        # Kick off the experiment ------------------------------------------
        payload: dict[str, Any] = {
            "project_id": project,
            "dataset_id": self._dataset_id,
            "dimensions": scores,
            "model": model,
            "name": self._experiment_name,
        }

        response = _post("/v1/experiments", json=payload)
        self._result: ExperimentResult = _parse_experiment(response)

        # Optionally block until finished ----------------------------------
        if wait and not self._result.is_complete:
            self._result = _poll_experiment(
                experiment_id=self._result.experiment_id,
                timeout=timeout,
            )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def experiment_id(self) -> str:
        return self._result.experiment_id

    @property
    def status(self) -> str:
        return self._result.status

    @property
    def score(self) -> float:
        return self._result.score

    @property
    def summary(self) -> dict[str, Any]:
        """Per-dimension average scores, e.g. ``{"visual_quality": 4.2}``."""
        return self._result.summary

    @property
    def result(self) -> ExperimentResult:
        """The full ``ExperimentResult`` object."""
        return self._result

    @property
    def scores(self) -> list[EvalResult]:
        """Individual per-item ``EvalResult`` objects."""
        return self._result.scores

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> ExperimentResult:
        """Fetch the latest state from the API (useful when ``wait=False``)."""
        data = _get(f"/v1/experiments/{self._result.experiment_id}")
        self._result = _parse_experiment(data)
        return self._result

    def __repr__(self) -> str:
        return (
            f"Eval(name={self._experiment_name!r}, model={self._model!r}, "
            f"status={self.status!r}, score={self.score})"
        )
