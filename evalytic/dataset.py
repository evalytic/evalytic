"""Dataset creation and management.

Datasets hold the "golden test set" of inputs that experiments run against.

Example:
    import evalytic
    evalytic.init(api_key="ek-...")

    ds = evalytic.init_dataset("my-project", "hero-prompts", pipeline_type="text2img")
    ds.insert({"prompt": "A cat wearing a top hat"})
    ds.insert({"prompt": "Sunset over Tokyo"}, expected={"style": "photorealistic"})
    dataset_id = ds.push()
"""

from __future__ import annotations

from typing import Any

from .client import _post


class Dataset:
    """A collection of evaluation inputs that can be pushed to the Evalytic API.

    Items are accumulated locally via ``insert()`` and then sent in a single
    batch with ``push()``.
    """

    def __init__(
        self,
        project_id: str,
        name: str,
        pipeline_type: str = "text2img",
    ) -> None:
        self.project_id = project_id
        self.name = name
        self.pipeline_type = pipeline_type
        self.dataset_id: str | None = None
        self._items: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(
        self,
        input: dict[str, Any],
        expected: dict[str, Any] | None = None,
    ) -> None:
        """Add a test-set item to the local buffer.

        Parameters
        ----------
        input:
            The input data for evaluation.  For *text2img* this typically
            contains a ``"prompt"`` key.  For *img2img* it contains
            ``"image"`` (URL) and ``"instruction"``.
        expected:
            Optional expected-output metadata used for reference comparison
            (e.g. expected style, expected objects present).
        """
        item: dict[str, Any] = {"input": input}
        if expected is not None:
            item["expected"] = expected
        self._items.append(item)

    def push(self) -> str:
        """Push the dataset to the Evalytic API.

        Returns
        -------
        str
            The server-assigned ``dataset_id``.

        Raises
        ------
        ValueError
            If the dataset has no items.
        RuntimeError
            If the dataset was already pushed.
        """
        if self.dataset_id is not None:
            raise RuntimeError(
                f"Dataset '{self.name}' has already been pushed "
                f"(dataset_id={self.dataset_id}). Create a new Dataset to "
                "push additional data."
            )

        if not self._items:
            raise ValueError(
                "Cannot push an empty dataset. Add items with .insert() first."
            )

        payload: dict[str, Any] = {
            "project_id": self.project_id,
            "name": self.name,
            "pipeline_type": self.pipeline_type,
            "items": self._items,
        }

        data = _post("/v1/datasets", json=payload)
        self.dataset_id = data.get("dataset_id", "")
        return self.dataset_id

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def is_pushed(self) -> bool:
        """Whether the dataset has been pushed to the API."""
        return self.dataset_id is not None

    @property
    def size(self) -> int:
        """Number of items in the local buffer."""
        return len(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        status = f"id={self.dataset_id}" if self.dataset_id else "local"
        return (
            f"Dataset(name={self.name!r}, pipeline={self.pipeline_type!r}, "
            f"items={len(self._items)}, {status})"
        )


def init_dataset(
    project_id: str,
    name: str,
    pipeline_type: str = "text2img",
) -> Dataset:
    """Create a new local dataset ready to accept items.

    Parameters
    ----------
    project_id:
        The project this dataset belongs to.
    name:
        Human-readable name for the dataset.
    pipeline_type:
        ``"text2img"`` or ``"img2img"``.

    Returns
    -------
    Dataset
    """
    return Dataset(
        project_id=project_id,
        name=name,
        pipeline_type=pipeline_type,
    )
