"""Tests for Dataset."""

import numpy as np
import pytest

from bo_arena.utils.dataset import Dataset


def _dataset(n: int, d: int = 3, m: int = 1, start: int = 0) -> Dataset:
    return Dataset(
        X=np.arange(start, start + n * d, dtype=np.float64).reshape(n, d),
        objectives=np.arange(start, start + n * m, dtype=np.float64).reshape(n, m),
    )


def test_properties() -> None:
    """The n and n_objectives properties should reflect the array shapes."""
    data = _dataset(n=4, m=2)
    assert data.n == 4
    assert data.n_objectives == 2


@pytest.mark.parametrize(
    ("X", "objectives"),
    [
        (np.zeros((4, 3)), np.zeros(4)),  # 1D objectives
        (np.zeros((4, 3)), np.zeros((5, 1))),  # mismatched N
    ],
)
def test_invalid_shapes_raise(X: np.ndarray, objectives: np.ndarray) -> None:
    """Invalid shapes should be rejected on construction."""
    with pytest.raises(ValueError):  # ruff: ignore[pytest-raises-too-broad]
        Dataset(X=X, objectives=objectives)


def test_add_concatenates_in_order() -> None:
    """Adding datasets should concatenate X and objectives, preserving order."""
    left, right = _dataset(n=2), _dataset(n=3, start=100)
    combined = left + right
    np.testing.assert_array_equal(combined.X, np.concatenate((left.X, right.X)))
    np.testing.assert_array_equal(
        combined.objectives, np.concatenate((left.objectives, right.objectives))
    )


def test_add_to_empty() -> None:
    """Adding to an empty dataset (as the Arena does) should return the other data."""
    empty = Dataset(X=np.empty((0, 3)), objectives=np.empty((0, 1)))
    data = _dataset(n=2)
    combined = empty + data
    np.testing.assert_array_equal(combined.X, data.X)
    np.testing.assert_array_equal(combined.objectives, data.objectives)


def test_add_mismatched_objectives_raises() -> None:
    """Datasets with different numbers of objectives cannot be combined."""
    with pytest.raises(ValueError, match="objectives"):
        _ = _dataset(n=2, m=1) + _dataset(n=2, m=2)


def test_constraints_default_to_empty() -> None:
    """Omitting constraints should give an unconstrained dataset."""
    data = _dataset(n=4)
    assert data.constraints == {}


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((4, 1)),  # 2D values
        np.zeros(5),  # mismatched N
    ],
)
def test_invalid_constraint_shapes_raise(values: np.ndarray) -> None:
    """Constraint values must be 1D with one value per input."""
    with pytest.raises(ValueError, match="Constraint 'g'"):
        Dataset(
            X=np.zeros((4, 3)), objectives=np.zeros((4, 1)), constraints={"g": values}
        )


def test_constraints_are_copied() -> None:
    """Later changes to the caller's mapping should not affect the dataset."""
    constraints = {"g": np.zeros(2)}
    data = Dataset(
        X=np.zeros((2, 3)), objectives=np.zeros((2, 1)), constraints=constraints
    )
    constraints["h"] = np.zeros(2)
    assert tuple(data.constraints) == ("g",)


def test_add_concatenates_constraints_by_name() -> None:
    """Adding datasets should concatenate each constraint by name, in left's order."""
    left = Dataset(
        X=np.zeros((2, 3)),
        objectives=np.zeros((2, 1)),
        constraints={"g": np.array([1.0, 2.0]), "h": np.array([3.0, 4.0])},
    )
    right = Dataset(
        X=np.zeros((1, 3)),
        objectives=np.zeros((1, 1)),
        constraints={"h": np.array([30.0]), "g": np.array([10.0])},
    )
    combined = left + right
    assert tuple(combined.constraints) == ("g", "h")
    np.testing.assert_array_equal(combined.constraints["g"], [1.0, 2.0, 10.0])
    np.testing.assert_array_equal(combined.constraints["h"], [3.0, 4.0, 30.0])


def test_add_mismatched_constraints_raises() -> None:
    """Datasets with different constraint names cannot be combined."""
    constrained = Dataset(
        X=np.zeros((2, 3)), objectives=np.zeros((2, 1)), constraints={"g": np.zeros(2)}
    )
    with pytest.raises(ValueError, match="constraints"):
        _ = _dataset(n=2) + constrained
