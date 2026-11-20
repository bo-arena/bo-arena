"""Class for storing datasets of observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jaxtyping import Float


@dataclass
class Dataset:
    """Base class for datasets.

    Args:
        X: input data.
        objectives: objective values, with one row per input and one column per
            objective.
        constraints: (Optional) raw black-box constraint values, as a mapping from
            constraint name to a 1-dimensional array with one value per input. Which
            values are feasible is defined by the problem's
            `ProblemMetadata.constraints`, matched by name.
    """

    X: Float[np.ndarray, "N ..."]
    objectives: Float[np.ndarray, "N n_objectives"]
    constraints: dict[str, Float[np.ndarray, " N"]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Checks that the shapes of $X$, objectives and constraints are compatible."""
        if self.objectives.ndim != 2:
            msg = f"Objectives must be a 2-dimensional array with shape (N, n_objectives). Got objectives.shape={self.objectives.shape}."  # ruff: ignore[line-too-long]
            raise ValueError(msg)

        if self.X.shape[0] != self.objectives.shape[0]:
            msg = f"Inputs, X, and objectives must have the same number of elements. Got X.shape={self.X.shape} and objectives.shape={self.objectives.shape}."  # ruff: ignore[line-too-long]
            raise ValueError(msg)

        # Copy so that later changes to the caller's mapping don't affect the dataset.
        self.constraints = dict(self.constraints)
        for name, values in self.constraints.items():
            if values.shape != (self.n,):
                msg = f"Constraint '{name}' must be a 1-dimensional array with one value per input. Got shape {values.shape} for X.shape={self.X.shape}."  # ruff: ignore[line-too-long]
                raise ValueError(msg)

    def __add__(self, other: Dataset) -> Dataset:
        """Combine two datasets.

        Order of left and right dataset is preserved.

        Args:
            other: Dataset to concatenate with this dataset.

        Returns:
            Concatenated dataset.

        Raises:
            ValueError: If the datasets have a different number of objectives or
                different constraint names.
        """
        if self.n_objectives != other.n_objectives:
            msg = f"Cannot combine datasets with {self.n_objectives} and {other.n_objectives} objectives."  # ruff: ignore[line-too-long]
            raise ValueError(msg)
        if set(self.constraints) != set(other.constraints):
            msg = f"Cannot combine datasets with constraints {sorted(self.constraints)} and {sorted(other.constraints)}."  # ruff: ignore[line-too-long]
            raise ValueError(msg)
        X = np.concatenate((self.X, other.X))
        objectives = np.concatenate((self.objectives, other.objectives))
        constraints = {
            name: np.concatenate((values, other.constraints[name]))
            for name, values in self.constraints.items()
        }
        return Dataset(X=X, objectives=objectives, constraints=constraints)

    @property
    def n(self) -> int:
        r"""Number of observations."""
        return self.X.shape[0]

    @property
    def n_objectives(self) -> int:
        r"""Number of objectives."""
        return self.objectives.shape[1]
