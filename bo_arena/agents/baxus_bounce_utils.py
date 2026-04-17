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
class BAxUSState:
    """BAxUS state used to track the trust region adaptive subspace expansion."""

    input_dim: int  # Corresponds to the original (input) dimensionality of the problem
    eval_budget: int
    new_bins_on_split: int = 3
    d_init: int = dataclasses.field(init=False)  # Note: Post-initialized
    target_dim: int = dataclasses.field(init=False)  # Note: Post-initialized
    n_splits: int = dataclasses.field(init=False)  # Note: Post-initialized
    length: float = 0.8
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        """Initialize the subspace dimensionality and split parameters."""
        self.n_splits = round(math.log(self.input_dim, self.new_bins_on_split + 1))
        self.d_init = int(
            1
            + np.argmin(
                np.abs(
                    (1 + np.arange(self.new_bins_on_split))
                    * (1 + self.new_bins_on_split) ** self.n_splits
                    - self.input_dim
                )
            )
        )
        self.target_dim = self.d_init

    @property
    def split_budget(self) -> int:
        """Calculate the evaluation budget allocated to each subspace split."""
        num = self.new_bins_on_split * self.eval_budget * self.target_dim
        den = self.d_init * (1 - (self.new_bins_on_split + 1) ** (self.n_splits + 1))
        return round(-1 * num / den)

    @property
    def failure_tolerance(self) -> int:
        """Calculate failures allowed before shrinking the trust region."""
        if self.target_dim == self.input_dim:
            return self.target_dim
        k = math.floor(math.log(self.length_min / self.length_init, 0.5))
        return min(self.target_dim, max(1, math.floor(self.split_budget / k)))


@dataclass
class BounceState:
    """Bounce state used to track the adaptive subspace expansion (continuous-only)."""

    # Problem and embedding configuration
    input_dim: int
    n_initial_points: int
    eval_budget: int
    new_bins_on_split: int = 2
    initial_target_dim: int = 2
    eval_budget_until_input_dim: int = 250

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
    terminated: bool = False

    def __post_init__(self) -> None:
        """Compute split schedule and per-tier budgets."""
        # Initialize dimensionality and budget parameters
        self.eval_budget_until_input_dim = max(
            self.n_initial_points + 1,
            min(self.eval_budget_until_input_dim, self.eval_budget),
        )
        self.d_init = min(self.initial_target_dim, self.input_dim)
        self.target_dim = self.d_init

        # Compute number of splits and optimize bin count
        self.n_splits = max(
            1,
            math.ceil(
                math.log(
                    max(self.input_dim, 2) / max(self.d_init, 1),
                    self.new_bins_on_split + 1,
                )
            ),
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
        self.split_budgets[self.input_dim] = self.split_budget_for(self.input_dim)

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
            self.n_splits = max(
                1,
                math.ceil(
                    math.log(self.input_dim / self.d_init, self.new_bins_on_split + 1)
                ),
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
            return max(1, min(10 * target_dim, self.eval_budget))
        num = self.new_bins_on_split * total_budget * target_dim
        den = self.d_init * (1 - (self.new_bins_on_split + 1) ** (self.n_splits + 1))
        split_budget = round(-num / den)
        return max(1, min(2**target_dim, split_budget))

    def reset_trust_region(self) -> None:
        """Reset trust region lengths after split or restart."""
        self.length_discrete_continuous = float(
            min(self.length_init_discrete, self.target_dim)
        )
        self.length_continuous = self.length_init_continuous
        self.terminated = False


def update_baxus_state(
    state: BAxUSState, y_next: Float[torch.Tensor, "n 1"]
) -> BAxUSState:
    """Update the state of the trust region based on the new function values.

    Note:
        This function modifies the `state` object in-place.

    Args:
        state: Current BAxUS state.
        y_next: New function values from the latest batch.

    Returns:
        Updated BAxUS state.
    """
    if max(y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True

    return state


def compute_adjustment_factor(
    state: BounceState, n_evals: int, batch_size: int
) -> float:
    """Calculate the geometric shrink factor for the trust region.

    Args:
        state: Current Bounce state.
        n_evals: Number of evaluations completed so far.
        batch_size: Size of the current batch.

    Returns:
        Geometric adjustment factor for the trust region length.
    """
    # Remaining evaluation budget for the current dimensionality tier
    remaining = state.split_budgets.get(
        state.target_dim, state.split_budget_for(state.target_dim)
    )
    remaining = min(remaining, state.eval_budget - n_evals)
    remaining = max(remaining, 1)

    # Geometric factor chosen so that length_discrete_continuous reaches
    # length_min_discrete in exactly `remaining` single-point steps
    factor = (state.length_min_discrete / state.length_discrete_continuous) ** (
        1.0 / remaining
    )
    factor = factor**batch_size
    return float(np.clip(factor, 1e-10, None))


def update_bounce_state(
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
            float(state.input_dim),  # length_max_discrete = dimensionality
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
