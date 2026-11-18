"""Utilities for black-box constraints."""

from collections.abc import Sequence

import numpy as np
from jaxtyping import Bool, Float

from bo_arena.problems.metadata import ConstraintSpec
from bo_arena.utils.dataset import Dataset


def is_feasible(
    data: Dataset, constraint_specs: Sequence[ConstraintSpec]
) -> Bool[np.ndarray, " N"]:
    """Determine which observations satisfy all of their constraints.

    An observation is feasible when every constraint value `c` satisfies
    `lower <= c <= upper` for the bounds of the spec with the same name, where `None`
    means unbounded on that side. NaN constraint values are treated as infeasible.
    With no constraints, every observation is feasible.

    Args:
        data: Dataset whose constraint values to check.
        constraint_specs: Constraint specifications (see
            `ProblemMetadata.constraints`), matched to the dataset's constraints by
            name.

    Returns:
        Boolean mask that is True for observations satisfying all constraints.

    Raises:
        ValueError: If the dataset's constraint names differ from the specs' names.
    """
    spec_names = {spec.name for spec in constraint_specs}
    if spec_names != set(data.constraints):
        msg = (
            f"Dataset has constraints {sorted(data.constraints)}, but the "
            f"constraint specs are for {sorted(spec_names)}."
        )
        raise ValueError(msg)
    feasible = np.ones(data.n, dtype=bool)
    for spec in constraint_specs:
        values = data.constraints[spec.name]
        lower = -np.inf if spec.lower is None else spec.lower
        upper = np.inf if spec.upper is None else spec.upper
        feasible &= (values >= lower) & (values <= upper)
    return feasible


def running_best_feasible(
    observations: Float[np.ndarray, " N"], feasible: Bool[np.ndarray, " N"]
) -> Float[np.ndarray, " N"]:
    """Compute the best feasible observation seen up to and including each index.

    Args:
        observations: Objective values, in evaluation order.
        feasible: Mask that is True for observations satisfying all constraints.

    Returns:
        Running maximum over feasible observations, NaN until the first feasible
        observation with a non-NaN objective.
    """
    # `fmax` (unlike `maximum`) returns the non-NaN operand when one is NaN, so a
    # single NaN objective cannot poison the rest of the running best.
    best = np.fmax.accumulate(np.where(feasible, observations, -np.inf))
    return np.where(best == -np.inf, np.nan, best)
