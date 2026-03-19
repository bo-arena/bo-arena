"""BAxUS (Bayesian Optimization with Adaptively Expanding Subspaces) agent."""

import dataclasses
import math
from dataclasses import dataclass

import botorch
import gpytorch
import numpy as np
import torch
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from jaxtyping import Float
from torch.quasirandom import SobolEngine

from bo_arena.agents.ask_tell_agent import AskTellAgent
from bo_arena.utils.dataset import Dataset


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


def update_state(state: BAxUSState, y_next: Float[torch.Tensor, "n 1"]) -> BAxUSState:
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

        # Set values that were move to zero in current row
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


class BAxUS(AskTellAgent):
    """Bayesian Optimization with Adaptively Expanding Subspaces (BAxUS) agent.

    This agent implements BAxUS as described in "Increasing the Scope as You Learn:
    Adaptive Bayesian Optimization in Nested Subspaces" by Papenmeier et al. (2022).
    BAxUS maintains a trust region in a low-dimensional subspace and adaptively
    increases the subspace dimensionality.

    Reference:
        Papenmeier, L., et al. (2022). Increasing the Scope as You Learn: Adaptive
        Bayesian Optimization in Nested Subspaces. Neural Information Processing Systems
        (NeurIPS).
    """

    def __init__(
        self,
        input_dim: int,
        acqf_opt_strategy: str = "default",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        use_dsp: bool = False,  # noqa: FBT002
        **kwargs: object,
    ) -> None:
        """Initialize the BAxUS agent.

        Args:
            input_dim: Dimensionality of the input space (assumed to be [0,1]^d).
            acqf_opt_strategy: Strategy for optimizing the acquisition function. Can be
              "default", "no_raasp", or "raasp". Setting "default" uses the same
              settings as in the original paper.
            dtype: The torch dtype to use for the model and data.
            device: The torch device to use for the model and data.
            use_dsp: If True, use the dimensionality-scaled prior (DSP) from Hvarfner
              et al. (2024) instead of the default BAxUS kernel and priors.
            kwargs: Additional arguments, including 'n_total_evaluations'.

        """
        self.dtype = dtype
        self.device = device
        self.input_dim = input_dim
        self.use_dsp = use_dsp
        self.n_total_evaluations: int | None = kwargs.get("n_total_evaluations")  # type: ignore[attr-defined]
        if acqf_opt_strategy not in {"default", "no_raasp", "raasp"}:
            msg = f"Unknown acqf_opt_strategy: {acqf_opt_strategy!r}"
            raise ValueError(msg)
        self.acqf_opt_strategy = acqf_opt_strategy

        self.train_X: torch.Tensor | None = None
        self.train_y = torch.empty(0, 1, dtype=dtype, device=device)
        self.model = None
        self.state: BAxUSState | None = None
        self._restart_buffer = torch.empty(
            0, self.input_dim, dtype=dtype, device=device
        )

        # Projection matrix
        self.S: Float[torch.Tensor, "target_dim input_dim"] | None = None

        # BAxUS assumes input space is [-1,1]^d
        self.bounds = torch.stack([
            -torch.ones(self.input_dim, dtype=self.dtype, device=self.device),
            torch.ones(self.input_dim, dtype=self.dtype, device=self.device),
        ])

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return "baxus_dsp" if self.use_dsp else "baxus"

    def _create_state(self) -> BAxUSState:
        """Create a new BAxUSState, validating that n_total_evaluations is set."""
        if self.n_total_evaluations is None:
            msg = (
                "n_total_evaluations is required for BAxUS. "
                "Pass it as a keyword argument to the constructor."
            )
            raise RuntimeError(msg)
        return BAxUSState(
            input_dim=self.input_dim, eval_budget=self.n_total_evaluations
        )

    def _fit_model(self) -> None:
        """Fit a SingleTaskGP model in the low-dimensional subspace.

        Uses the BAxUS Matérn kernel with explicit constraints as in the original paper
        when use_dsp=False, or the BoTorch default SingleTaskGP with DSP priors when
        use_dsp=True.
        """
        if self.use_dsp:
            self.model = SingleTaskGP(self.train_X, self.train_y)
        else:
            likelihood = GaussianLikelihood(noise_constraint=Interval(5e-4, 0.2))
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=self.train_X.shape[-1],
                    lengthscale_constraint=Interval(0.005, 10.0),
                ),
                outputscale_constraint=Interval(0.05, 20.0),
            )
            self.model = SingleTaskGP(
                self.train_X,
                self.train_y,
                covar_module=covar_module,
                likelihood=likelihood,
            )
            # Initialize model hypers
            self.model.initialize(**{
                "covar_module.outputscale": 1.0,
                "covar_module.base_kernel.lengthscale": 0.5,
                "likelihood.noise": 0.005,
            })

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        with gpytorch.settings.max_cholesky_size(float("inf")):  # Always use Cholesky
            try:
                fit_gpytorch_mll(mll)

            except ModelFittingError:
                # Right after increasing the target dim, the covariance matrix may
                # become indefinite, and thus Cholesky decomposition might fail.
                self.model.train()
                optimizer = torch.optim.Adam(
                    [{"params": self.model.parameters()}], lr=0.1
                )
                for _ in range(100):
                    optimizer.zero_grad()
                    output = self.model(self.train_X)
                    loss = -mll(output, self.model.train_targets.flatten())
                    loss.backward()
                    optimizer.step()
                self.model.eval()

    def _get_trust_region_bounds(
        self,
        center: Float[torch.Tensor, " target_dim"],
        length: float,
    ) -> tuple[Float[torch.Tensor, " target_dim"], Float[torch.Tensor, " target_dim"]]:
        """Get the trust region bounds centered around the best point.

        Args:
            center: Center point of the trust region (best point found so far).
            length: Length of the trust region.

        Returns:
            A tuple of (tr_lb, tr_ub) tensors in [-1,1]^d space.
        """
        # Scale the TR to be proportional to the lengthscales
        if self.use_dsp:
            # With DSP enabled, there is no ScaleKernel wrapping the base kernel
            lengthscale = self.model.covar_module.lengthscale
        else:
            lengthscale = self.model.covar_module.base_kernel.lengthscale
        weights = lengthscale.squeeze().detach()
        if weights.ndim > 0:  # target_dim might be 1
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

        # Create trust region bounds in [-1,1]^d space with lengthscale scaling
        tr_lb = torch.clamp(center - weights * length, -1.0, 1.0)
        tr_ub = torch.clamp(center + weights * length, -1.0, 1.0)

        return tr_lb, tr_ub

    def generate_initial_points(self, n: int, seed: int) -> Float[np.ndarray, "N D"]:
        """Generate initial points in the low-dimensional subspace and project up.

        Initializes the BAxUS state and embedding matrix early, generates Sobol points
        in [-1,1]^target_dim, then projects them to the full input space.

        Args:
            n: Number of initial points to generate.
            seed: Random seed for reproducibility.

        Returns:
            Initial points in [0,1]^d for the Arena to evaluate.
        """
        # Initialize state and embedding matrix early
        self.state = self._create_state()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.S = embedding_matrix(
            self.state.input_dim,
            self.state.d_init,
            generator=generator,
            dtype=self.dtype,
            device=self.device,
        )

        # Generate Sobol points in [-1,1]^target_dim
        sobol = SobolEngine(self.state.target_dim, scramble=True, seed=seed)
        X_target = 2 * sobol.draw(n).to(dtype=self.dtype, device=self.device) - 1

        # Project X to full-dimensional input space, i.e. [-1,1]^d
        X_full = X_target @ self.S

        # BO-Arena assumes input space is [0,1]^d
        X_normalized = normalize(X=X_full, bounds=self.bounds)

        return X_normalized.detach().cpu().numpy()

    def tell_initial_data(self, data: Dataset, seed: int) -> None:  # noqa: ARG002
        """Provide initial observations to the agent and fit the SingleTaskGP model.

        Args:
            data: Initial observations to provide to the agent.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.

        Raises:
            RuntimeError: If `generate_initial_points` has not been called first.
        """
        if self.state is None:
            msg = (
                "BAxUS state has not been initialized. "
                "generate_initial_points must be called before tell_initial_data."
            )
            raise RuntimeError(msg)

        # BAxUS assumes input space is [-1,1]^d
        X_normalized = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        X_unnormalized = unnormalize(X=X_normalized, bounds=self.bounds)

        # Project X to low-dimensional input space
        self.train_X = torch.linalg.lstsq(self.S.t(), X_unnormalized.t()).solution.t()
        self.train_y = torch.tensor(
            data.y, dtype=self.dtype, device=self.device
        ).unsqueeze(-1)

        self.n_init = self.train_X.shape[0]  # Used for BAxUS restarts

        # Fit model, but disable input scaling checks as we normalize to [-1,1]^d
        with botorch.settings.validate_input_scaling(False):
            self._fit_model()

    def tell(self, data: Dataset, seed: int) -> None:
        """Update the agent with new observations and refit the SingleTaskGP model.

        Args:
            data: Observations corresponding to the points requested in the previous
                `ask` call.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.

        Raises:
            RuntimeError: If the BAxUS state has not been initialized with
                `tell_initial_data`.
        """
        if self.state is None:
            msg = "BAxUS state has not been initialized. Call tell_initial_data first."
            raise RuntimeError(msg)

        # BAxUS assumes input space is [-1,1]^d
        X_normalized = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        X_unnormalized = unnormalize(X=X_normalized, bounds=self.bounds)

        # Project X to low-dimensional input space
        new_X = torch.linalg.lstsq(self.S.t(), X_unnormalized.t()).solution.t()
        new_y = torch.tensor(data.y, dtype=self.dtype, device=self.device).unsqueeze(-1)

        self.state = update_state(self.state, new_y)

        self.train_X = torch.cat((self.train_X, new_X), dim=0)
        self.train_y = torch.cat([self.train_y, new_y])

        if self.state.restart_triggered:
            if self.state.target_dim < self.state.input_dim:
                # Increase target space
                generator = torch.Generator(device=self.device).manual_seed(seed)
                self.S, self.train_X = increase_embedding_and_observations(
                    self.S,
                    self.train_X,
                    self.state.new_bins_on_split,
                    generator=generator,
                )
                self.state.target_dim = len(self.S)

                # Partially reset trust region
                self.state.length = self.state.length_init
                self.state.failure_counter = 0
                self.state.success_counter = 0
                self.state.restart_triggered = False

            elif self.state.target_dim == self.state.input_dim:
                # Completely reset trust region
                self.state = self._create_state()
                self.state.target_dim = self.input_dim  # End of line 16 in Algorithm 1
                generator = torch.Generator(device=self.device).manual_seed(seed)
                self.S = embedding_matrix(
                    input_dim=self.state.input_dim,
                    target_dim=self.state.input_dim,
                    generator=generator,
                    dtype=self.dtype,
                    device=self.device,
                )

                # Purge local data
                self.train_X = torch.empty(
                    0, self.state.target_dim, dtype=self.dtype, device=self.device
                )
                self.train_y = torch.empty(0, 1, dtype=self.dtype, device=self.device)
                self.model = None

                # Generate Sobol points in [-1,1]^target_dim
                sobol = SobolEngine(self.state.target_dim, scramble=True, seed=seed)
                X_target = (
                    2 * sobol.draw(self.n_init).to(dtype=self.dtype, device=self.device)
                    - 1
                )

                # Project X to full-dimensional input space, i.e. [-1,1]^d
                X_full = X_target @ self.S

                # BO-Arena assumes input space is [0,1]^d
                self._restart_buffer = normalize(X=X_full, bounds=self.bounds)

            else:
                msg = (
                    f"Impossible state: target_dim={self.state.target_dim} "
                    f"> dim={self.state.input_dim}"
                )
                raise RuntimeError(msg)

        # No need to fit model when init buffer is still being unrolled
        if self._restart_buffer.shape[0] == 0:
            # Fit model, but disable input scaling checks as we normalize to [-1,1]^d
            with botorch.settings.validate_input_scaling(False):
                self._fit_model()

    def ask(self, n: int, seed: int) -> Float[np.ndarray, "{n} D"]:
        """Select the next points to evaluate using Thompson Sampling in the subspace.

        This method uses Thompson Sampling (MaxPosteriorSampling) to generate
        candidates within a trust region in a dynamic low-dimensional subspace.

        Args:
            n: Number of points to select (used for batch BO).
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.

        Returns:
            Points selected by the agent for evaluation (in the full-dimensional space).

        Raises:
            RuntimeError: If the BAxUS state has not been initialized with
                `tell_initial_data`.
        """
        if self.state is None:
            msg = "BAxUS state has not been initialized. Call tell_initial_data first."
            raise RuntimeError(msg)

        # Consume buffer: Feed pre-generated points sequentially
        if self._restart_buffer.shape[0] > 0:
            batch_n = min(n, self._restart_buffer.shape[0])
            samples = self._restart_buffer[:batch_n]
            self._restart_buffer = self._restart_buffer[batch_n:]
            return samples.cpu().numpy()

        dim = self.train_X.shape[-1]

        # Find the best point (center of trust region)
        best_idx = self.train_y.argmax()
        x_center = self.train_X[best_idx].clone()

        # Get trust region bounds (in [-1,1]^d space)
        tr_lb, tr_ub = self._get_trust_region_bounds(x_center, self.state.length)

        # Use Sobol to generate perturbations
        if self.acqf_opt_strategy == "default":
            n_candidates = min(100 * dim, 5000)
        elif self.acqf_opt_strategy in {"no_raasp", "raasp"}:
            # For "no_raasp" and "raasp" we fix the number of candidates
            n_candidates = 5000
        sobol = SobolEngine(dim, scramble=True, seed=seed)
        sobol_samples = sobol.draw(n_candidates).to(
            dtype=self.dtype, device=self.device
        )
        sobol_samples = tr_lb + (tr_ub - tr_lb) * sobol_samples

        if self.acqf_opt_strategy in {"default", "raasp"}:
            # Candidates are formed by perturbing (a subset of dimensions of) the
            # center point within the trust region.
            prob_perturb = min(20.0 / dim, 1.0)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            mask = (
                torch.rand(
                    n_candidates,
                    dim,
                    dtype=self.dtype,
                    device=self.device,
                    generator=generator,
                )
                <= prob_perturb
            )
            # Ensure at least one dimension is perturbed for each candidate
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            if len(ind) > 0:
                mask[
                    ind,
                    torch.randint(
                        0,
                        dim,
                        size=(len(ind),),
                        device=self.device,
                        generator=generator,
                    ),
                ] = 1

            # Create candidate points from the center with perturbations
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = sobol_samples[mask]
        else:
            # For "no_raasp", candidates are sampled uniformly within the trust region
            X_cand = sobol_samples

        # Sample on the candidate points using Thompson Sampling
        with (
            botorch.settings.validate_input_scaling(False),
            gpytorch.settings.max_cholesky_size(float("inf")),
        ):  # Always use Cholesky
            thompson_sampling = MaxPosteriorSampling(
                model=self.model, replacement=False
            )
            with torch.no_grad():  # We don't need gradients when using TS
                X_cand = X_cand.unsqueeze(0)
                candidates = thompson_sampling(X_cand, num_samples=n).squeeze(0)

        # Project X to full-dimensional input space, i.e. [-1,1]^d
        X_next = candidates @ self.S

        # BO-Arena assumes input space is [0,1]^d
        X_next_normalized = normalize(X=X_next, bounds=self.bounds)

        return X_next_normalized.detach().cpu().numpy()
