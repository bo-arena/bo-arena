"""Tests for bo_arena.utils.acqf_opt."""

import pytest
import torch

from bo_arena.utils.acqf_opt import get_candidates


def _make_bounds(
    input_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (best_x, lb, ub) for an asymmetric, non-unit hypercube."""
    lb = torch.full((input_dim,), 0.2, dtype=torch.float64)
    ub = torch.full((input_dim,), 0.7, dtype=torch.float64)
    best_x = torch.full((input_dim,), 0.35, dtype=torch.float64)
    return best_x, lb, ub


@pytest.mark.parametrize("method", ["raasp", "sobol"])
def test_get_candidates_shape(method: str) -> None:
    """Output shape should be (n_candidates, input_dim)."""
    input_dim = 8
    n_candidates = 256
    best_x, lb, ub = _make_bounds(input_dim)

    X = get_candidates(
        best_x=best_x,
        lb=lb,
        ub=ub,
        n_candidates=n_candidates,
        seed=0,
        method=method,
        dtype=torch.float64,
        device="cpu",
    )

    assert X.shape == (n_candidates, input_dim)


@pytest.mark.parametrize("method", ["raasp", "sobol"])
def test_get_candidates_in_bounds(method: str) -> None:
    """All returned coordinates should lie within [lb, ub]."""
    input_dim = 8
    best_x, lb, ub = _make_bounds(input_dim)

    X = get_candidates(
        best_x=best_x,
        lb=lb,
        ub=ub,
        n_candidates=1024,
        seed=1,
        method=method,
        dtype=torch.float64,
        device="cpu",
    )

    assert torch.all(lb <= X)
    assert torch.all(ub >= X)


@pytest.mark.parametrize("method", ["raasp", "sobol"])
def test_get_candidates_deterministic(method: str) -> None:
    """Two calls with the same seed should produce identical outputs."""
    best_x, lb, ub = _make_bounds(8)

    kwargs = {
        "best_x": best_x,
        "lb": lb,
        "ub": ub,
        "n_candidates": 256,
        "seed": 42,
        "method": method,
        "dtype": torch.float64,
        "device": "cpu",
    }

    X1 = get_candidates(**kwargs)
    X2 = get_candidates(**kwargs)

    assert torch.equal(X1, X2)


@pytest.mark.parametrize("method", ["raasp", "sobol"])
def test_get_candidates_different_seeds(method: str) -> None:
    """Two calls with different seeds should produce differing outputs."""
    best_x, lb, ub = _make_bounds(8)

    common = {
        "best_x": best_x,
        "lb": lb,
        "ub": ub,
        "n_candidates": 256,
        "method": method,
        "dtype": torch.float64,
        "device": "cpu",
    }

    X1 = get_candidates(seed=42, **common)
    X2 = get_candidates(seed=43, **common)

    assert not torch.equal(X1, X2)


def test_get_candidates_sobol_ignores_best_x() -> None:
    """For method="sobol", best_x should not affect the output."""
    _, lb, ub = _make_bounds(8)
    best_x_a = torch.full((8,), 0.3, dtype=torch.float64)
    best_x_b = torch.full((8,), 0.6, dtype=torch.float64)

    common = {
        "lb": lb,
        "ub": ub,
        "n_candidates": 128,
        "seed": 99,
        "method": "sobol",
        "dtype": torch.float64,
        "device": "cpu",
    }

    X_a = get_candidates(best_x=best_x_a, **common)
    X_b = get_candidates(best_x=best_x_b, **common)

    assert torch.equal(X_a, X_b)


def test_get_candidates_invalid_method_raises() -> None:
    """An unknown method should raise ValueError."""
    best_x, lb, ub = _make_bounds(4)

    with pytest.raises(ValueError, match="Unknown candidate generation method"):
        get_candidates(
            best_x=best_x,
            lb=lb,
            ub=ub,
            n_candidates=16,
            seed=0,
            method="not_a_method",
            dtype=torch.float64,
            device="cpu",
        )
