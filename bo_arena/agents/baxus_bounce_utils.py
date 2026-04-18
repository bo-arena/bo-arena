"""Shared utilities for BAxUS and Bounce agents.

This module contains shared state dataclasses, helper functions, and state
management utilities used by both the BAxUS and Bounce agents.
"""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float


@dataclass
class BounceState:
    """Bounce state used to track the adaptive subspace expansion (continuous-only)."""

    # Problem and embedding configuration
    input_dim: int
    n_initial_points: int
    eval_budget: int  # Total number of evaluations (including initial points)
    # allocated to the entire run
    new_bins_on_split: int = 2
    initial_target_dim: int = 2
    eval_budget_until_input_dim: int = 250  # TODO(Thomas-Christie): a sensible way of
    # setting this (and some of the properties above) should be considered when we
    # release bo-arena.

    # Computed embedding and dimensionality state (post-initialized)
    d_init: int = dataclasses.field(init=False)
    target_dim: int = dataclasses.field(init=False)
    n_splits: int = dataclasses.field(init=False)

    # Trust region parameters
    length_continuous: float = 0.8
    length_init_continuous: float = 0.8
    length_min_continuous: float = 0.5**7
    length_max_continuous: float = 1.6
    # Retained from the mixed-space Bounce paper despite the continuous-only setting:
    # drives the geometric adjustment factor via the ratio
    # (length_min_discrete / length_discrete_continuous).
    length_discrete_continuous: float = dataclasses.field(init=False)
    length_init_discrete: int = 10
    length_min_discrete: int = 1

    # Optimization state
    split_budgets: dict[int, int] = dataclasses.field(init=False)
    n_evals: int = 0  # Stores total number of evaluations (including initial points),
    # managed by Bounce agent
    terminated: bool = False

    def __post_init__(self) -> None:
        """Compute split schedule and per-tier budgets."""
        if self.eval_budget_until_input_dim > self.eval_budget:
            msg = (
                f"eval_budget_until_input_dim ({self.eval_budget_until_input_dim}) "
                f"cannot exceed eval_budget ({self.eval_budget})"
            )
            raise ValueError(msg)
        if self.eval_budget_until_input_dim <= self.n_initial_points:
            msg = (
                f"eval_budget_until_input_dim ({self.eval_budget_until_input_dim}) "
                f"must exceed n_initial_points ({self.n_initial_points}) to leave "
                f"at least one evaluation for the adaptive phase"
            )
            raise ValueError(msg)

        self.d_init = min(self.initial_target_dim, self.input_dim)
        self.target_dim = self.d_init

        # Compute number of splits and optimize bin count
        self.n_splits = math.ceil(
            math.log(
                self.input_dim / self.d_init,
                self.new_bins_on_split + 1,
            )
        )
        self._adjust_number_bins_on_split()

        # Initialize trust region and compute per-tier budgets
        self.length_discrete_continuous = float(
            min(self.length_init_discrete, self.target_dim)
        )

        # Allocate evaluation budget to each dimensionality tier
        tiers = [
            int(self.d_init * (self.new_bins_on_split + 1) ** k)
            for k in range(self.n_splits + 1)
        ]
        self.split_budgets = {t: self.split_budget_for(t) for t in tiers}

    def _adjust_number_bins_on_split(self) -> None:
        """Pick ``new_bins_on_split`` that best reaches input_dim after n_splits splits.

        The default value may not evenly reach input_dim via
        ``d_init * (new_bins_on_split + 1)^n_splits``. This corrects for that by
        searching over candidate bin counts and selecting the one with the smallest
        overshoot/undershoot.
        """
        if self.d_init >= self.input_dim:
            return
        upper = max(2, math.floor(math.log2(max(self.input_dim, 2))))
        possible = np.arange(1, upper)
        if possible.size == 0:
            possible = np.array([1])
        errs = np.abs(self.d_init * (1 + possible) ** self.n_splits - self.input_dim)
        best = int(possible[int(np.argmin(errs))])
        if best != self.new_bins_on_split:
            self.new_bins_on_split = best
            self.n_splits = math.ceil(
                math.log(self.input_dim / self.d_init, self.new_bins_on_split + 1)
            )

    def split_budget_for(self, target_dim: int) -> int:
        """Calculate per-tier evaluation budget.

        Args:
            target_dim: Target dimensionality for this tier.

        Returns:
            Evaluation budget allocated to this tier.
        """
        total_budget = self.eval_budget_until_input_dim - self.n_initial_points
        if target_dim >= self.input_dim:
            return min(10 * target_dim, self.eval_budget - self.n_evals)
        num = self.new_bins_on_split * total_budget * target_dim
        den = self.d_init * (1 - (self.new_bins_on_split + 1) ** (self.n_splits + 1))
        split_budget = round(-num / den)
        return min(2**target_dim, split_budget)

    def reset_trust_region(self) -> None:
        """Reset trust region lengths after split or restart."""
        self.length_discrete_continuous = float(
            min(self.length_init_discrete, self.target_dim)
        )
        self.length_continuous = self.length_init_continuous
        self.terminated = False


def compute_adjustment_factor(state: BounceState, batch_size: int) -> float:
    """Calculate the geometric shrink factor for the trust region.

    Args:
        state: Current Bounce state.
        batch_size: Size of the current batch.

    Returns:
        Geometric adjustment factor for the trust region length.
    """
    # Remaining evaluation budget for the current dimensionality tier
    remaining = state.split_budgets[state.target_dim]
    remaining = min(remaining, state.eval_budget - state.n_evals)
    remaining = max(remaining, 1)

    # Geometric factor chosen so that length_discrete_continuous reaches
    # length_min_discrete in exactly `remaining` single-point steps
    factor = (state.length_min_discrete / state.length_discrete_continuous) ** (
        1.0 / remaining
    )
    factor = factor**batch_size
    return float(np.clip(factor, a_min=1e-10, a_max=None))


def update_bounce_trust_region_state(
    state: BounceState,
    y_next: Float[torch.Tensor, "n 1"],
    y_best_prev: float,
    adjustment_factor: float,
) -> BounceState:
    """Update the trust region based on observed improvement.

    Shrinks or expands the trust region length based on whether improvement
    was observed, and marks termination when the TR shrinks below minimum length.

    Args:
        state: Current Bounce state.
        y_next: New function values from the latest batch.
        y_best_prev: Best observed value prior to this batch.
        adjustment_factor: Geometric factor for TR adjustment.

    Returns:
        Updated Bounce state.
    """
    # Determine if improvement was observed
    y_best_new = float(torch.max(y_next).item())
    improved = y_best_new > y_best_prev + 1e-3 * math.fabs(y_best_prev)

    # Shrink or expand trust region lengths geometrically
    if not improved:
        state.length_discrete_continuous *= adjustment_factor
        state.length_continuous *= adjustment_factor
    else:
        state.length_discrete_continuous = min(
            state.length_discrete_continuous / adjustment_factor,
            float(state.target_dim),
        )
        state.length_continuous = min(
            state.length_continuous / adjustment_factor,
            state.length_max_continuous,
        )

    # Check termination: TR has shrunk to minimum size
    discrete_done = state.length_discrete_continuous - 1e-4 <= state.length_min_discrete
    continuous_done = state.length_continuous - 1e-4 <= state.length_min_continuous
    if discrete_done or continuous_done:
        state.terminated = True

    return state


def embedding_matrix(
    input_dim: int,
    target_dim: int,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Float[torch.Tensor, "target_dim input_dim"]:
    """Create a sparse embedding matrix for dimensionality reduction.

    This matrix projects points from a high-dimensional space to a lower-dimensional
    subspace using a hashing-based approach.

    Args:
        input_dim: The dimensionality of the original input space.
        target_dim: The dimensionality of the target subspace.
        generator: Optional random number generator for reproducibility.
        dtype: The torch dtype for the output tensor.
        device: The torch device for the output tensor.

    Returns:
        A (target_dim, input_dim) tensor representing the embedding matrix.
    """
    # Return identity matrix if target size greater than input size
    if target_dim >= input_dim:
        return torch.eye(input_dim, dtype=dtype, device=device)

    # Add 1 to indices for padding column in matrix
    input_dims_perm = torch.randperm(input_dim, generator=generator, device=device) + 1

    # Split dims into almost equally-sized bins
    bins = torch.tensor_split(input_dims_perm, target_dim)

    # Zero pad bins, the index 0 will be cut off later
    bins = torch.nn.utils.rnn.pad_sequence(bins, batch_first=True)

    # Add one extra column for padding
    mtrx = torch.zeros((target_dim, input_dim + 1), dtype=dtype, device=device)

    # Fill mask with random +/- 1 at indices
    mtrx = mtrx.scatter_(
        1,
        bins,
        (
            2
            * torch.randint(
                2, (target_dim, input_dim), generator=generator, device=device
            )
            - 1
        ).to(mtrx.dtype),
    )

    # Cut off index zero as this corresponds to zero padding
    return mtrx[:, 1:]


def project_up(
    X_target_01: Float[torch.Tensor, "N target_dim"],
    S: Float[torch.Tensor, "target_dim input_dim"],
) -> Float[torch.Tensor, "N input_dim"]:
    """Project low-dim points up to the full input space via the embedding ``S``.

    The projection is performed in the ``[-1, 1]`` space where ``S`` is defined: input
    points are first rescaled from ``[0, 1]^target_dim`` to ``[-1, 1]^target_dim``,
    multiplied by ``S``, then rescaled back to ``[0, 1]^input_dim``.

    Args:
        X_target_01: A batch of points in ``[0, 1]^target_dim``.
        S: The count-sketch embedding matrix.

    Returns:
        A tensor of projected points in ``[0, 1]^input_dim``.
    """
    X_target = (2.0 * X_target_01) - 1.0
    X_full = X_target @ S
    return (X_full + 1.0) / 2.0


def project_down(
    X_full_01: Float[torch.Tensor, "N input_dim"],
    S: Float[torch.Tensor, "target_dim input_dim"],
) -> Float[torch.Tensor, "N target_dim"]:
    """Project full-dim points down to the target subspace via least-squares.

    Inverse of :func:`project_up`. Points are rescaled to ``[-1, 1]^input_dim``,
    solved against ``S.T`` in the least-squares sense, and rescaled back to
    ``[0, 1]^target_dim``. Recovery is exact (up to floating-point noise) when the
    input originated from ``project_up`` because ``S`` has full row rank; for
    arbitrary full-dim inputs, ``lstsq`` silently returns a min-norm pseudo-inverse
    solution rather than an exact recovery.

    Args:
        X_full_01: A batch of points in ``[0, 1]^input_dim``,
            assumed to have originated from :func:`project_up` with the same ``S``.
        S: The count-sketch embedding matrix.

    Returns:
        A tensor of projected points in ``[0, 1]^target_dim``.
    """
    X_full = (2.0 * X_full_01) - 1.0
    X_target = torch.linalg.lstsq(S.t(), X_full.t()).solution.t()
    return (X_target + 1.0) / 2.0


def increase_embedding_and_observations(
    S: Float[torch.Tensor, "target_dim input_dim"],
    X: Float[torch.Tensor, "n_obs target_dim"],
    n_new_bins: int,
    generator: torch.Generator | None = None,
) -> tuple[
    Float[torch.Tensor, "new_target_dim input_dim"],
    Float[torch.Tensor, "n_obs new_target_dim"],
]:
    """Increase the subspace dimensionality by splitting existing bins.

    This function implements the adaptive subspace expansion by splitting each row of
    the embedding matrix into multiple new rows, effectively increasing the target
    dimensionality while preserving the mapping for existing observations.

    Args:
        S: The current embedding matrix of shape (target_dim, input_dim).
        X: The current observations in the subspace.
        n_new_bins: The number of new bins to create from each existing bin.
        generator: Optional random number generator for reproducibility.

    Returns:
        A tuple (S_update, X_update) where:
            S_update: The expanded embedding matrix.
            X_update: The updated observations in the new subspace.

    Raises:
        ValueError: If the number of columns in X does not match the number of rows in
            S, indicating that the observations do not lie in the current subspace.
    """
    if X.size(1) != S.size(0):
        msg = (
            f"Observations don't lie in row space of S: "
            f"X.size(1)={X.size(1)}, S.size(0)={S.size(0)}"
        )
        raise ValueError(msg)

    S_update = S.clone()
    X_update = X.clone()

    for row_idx in range(len(S)):
        row = S[row_idx]
        idxs_non_zero = torch.nonzero(row)
        idxs_non_zero = idxs_non_zero[
            torch.randperm(len(idxs_non_zero), generator=generator, device=S.device)
        ].reshape(-1)

        if len(idxs_non_zero) <= 1:
            continue

        non_zero_elements = row[idxs_non_zero].reshape(-1)

        # Number of new bins is always less or equal than the contributing input dims in
        # the row minus one
        n_row_bins = min(n_new_bins + 1, len(idxs_non_zero))

        # The dims in the first bin won't be moved
        new_bins = torch.tensor_split(idxs_non_zero + 1, n_row_bins)[1:]
        elements_to_move = torch.tensor_split(non_zero_elements, n_row_bins)[1:]

        # Pad the tuples of bins with zeros to apply _scatter
        new_bins_padded = torch.nn.utils.rnn.pad_sequence(new_bins, batch_first=True)
        els_to_move_padded = torch.nn.utils.rnn.pad_sequence(
            elements_to_move, batch_first=True
        )

        # Submatrix to stack on S_update
        S_stack = torch.zeros(
            (n_row_bins - 1, len(row) + 1), dtype=S.dtype, device=S.device
        )

        # Fill with old values (add 1 to indices for padding column)
        S_stack = S_stack.scatter_(
            1, new_bins_padded, els_to_move_padded.to(S_stack.dtype)
        )

        # Set values that were moved to zero in current row
        S_update[row_idx, torch.hstack(new_bins) - 1] = 0

        # Repeat observations for row at the end of X (column-wise)
        X_update = torch.hstack((
            X_update,
            X[:, row_idx].reshape(-1, 1).repeat(1, len(new_bins)),
        ))

        # Stack onto S_update except for padding column
        S_update = torch.vstack((
            S_update,
            S_stack[:, 1:],
        ))

    return S_update, X_update
