"""TuRBO (Trust Region Bayesian Optimization) agent."""

import dataclasses
import math
from dataclasses import dataclass

import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from jaxtyping import Float
from torch.quasirandom import SobolEngine

from bo_arena.agents.ask_tell_agent import AskTellAgent
from bo_arena.utils.dataset import Dataset


@dataclass
class TuRBOState:
    """TuRBO state used to track the recent history of the trust region."""

    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = dataclasses.field(init=False)  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        """Post-initialize the state of the trust region."""
        # See: https://github.com/uber-research/TuRBO/blob/master/turbo/turbo_1.py#L106
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, self.dim / self.batch_size)
        )


def update_state(state: TuRBOState, y_next: Float[torch.Tensor, "n 1"]) -> TuRBOState:
    """Update the state of the trust region based on the new function values.

    Note:
        This function modifies the `state` object in-place.

    Args:
        state: Current TuRBO state.
        y_next: New function values from the latest batch.

    Returns:
        Updated TuRBO state.
    """
    # See: https://github.com/uber-research/TuRBO/blob/master/turbo/turbo_1.py#L138
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


class TuRBO(AskTellAgent):
    """Trust Region Bayesian Optimization (TuRBO-1) agent.

    This agent implements TuRBO-1 as described in "Scalable Global Optimization via
    Local Bayesian Optimization" by Eriksson et al. (2019). TuRBO maintains a trust
    region around the best point found so far and adaptively expands or contracts it
    based on success/failure.

    Reference:
        Eriksson, D., et al. (2019). Scalable Global Optimization via Local Bayesian
        Optimization. Neural Information Processing Systems (NeurIPS).
    """

    def __init__(
        self,
        input_dim: int,
        acqf_opt_strategy: str = "default",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        use_dsp: bool = False,  # noqa: FBT002
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Initialize the TuRBO agent.

        Args:
            input_dim: Dimensionality of the input space (assumed to be [0,1]^d).
            acqf_opt_strategy: Strategy for optimizing the acquisition function. Can be
              "default", "no_raasp", or "raasp". Setting "default" uses the same
              settings as in the original paper.
            dtype: The torch dtype to use for the model and data.
            device: The torch device to use for the model and data.
            use_dsp: If True, use the dimensionality-scaled prior (DSP) from Hvarfner
              et al. (2024) instead of the default TuRBO kernel and priors.
            kwargs: Additional arguments.

        Raises:
            ValueError: If an unknown acqf_opt_strategy is given.
        """
        self.dtype = dtype
        self.device = device
        self.input_dim = input_dim
        self.use_dsp = use_dsp
        if acqf_opt_strategy not in {"default", "no_raasp", "raasp"}:
            msg = f"Unknown acqf_opt_strategy: {acqf_opt_strategy!r}"
            raise ValueError(msg)
        self.acqf_opt_strategy = acqf_opt_strategy

        self.train_X = torch.empty(0, self.input_dim, dtype=dtype, device=device)
        self.train_y = torch.empty(0, 1, dtype=dtype, device=device)
        self.model = None
        self.state: TuRBOState | None = None
        self._restart_buffer = torch.empty(
            0, self.input_dim, dtype=dtype, device=device
        )

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return "turbo_dsp" if self.use_dsp else "turbo"

    def _fit_model(self) -> None:
        """Fit a SingleTaskGP model using ExactMarginalLogLikelihood.

        Uses the TuRBO Matérn kernel with explicit constraints as in the original paper
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
                    ard_num_dims=self.input_dim,
                    lengthscale_constraint=Interval(0.005, 2.0),
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
            fit_gpytorch_mll(mll)

    def _get_trust_region_bounds(
        self,
        center: Float[torch.Tensor, " D"],
        length: float,
    ) -> tuple[Float[torch.Tensor, " D"], Float[torch.Tensor, " D"]]:
        """Get the trust region bounds centered around the best point.

        Args:
            center: Center point of the trust region (best point found so far).
            length: Length of the trust region.

        Returns:
            A tuple of (tr_lb, tr_ub) tensors in [0,1]^d space.
        """
        # Scale the TR to be proportional to the lengthscales
        if self.use_dsp:
            # With DSP enabled, there is no ScaleKernel wrapping the base kernel
            lengthscale = self.model.covar_module.lengthscale
        else:
            lengthscale = self.model.covar_module.base_kernel.lengthscale
        weights = lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

        # Create trust region bounds in [0,1]^d space with lengthscale scaling
        tr_lb = torch.clamp(center - weights * length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(center + weights * length / 2.0, 0.0, 1.0)

        return tr_lb, tr_ub

    def tell_initial_data(self, data: Dataset, seed: int) -> None:  # noqa: ARG002
        """Provide initial observations to the agent and fit the SingleTaskGP model.

        Args:
            data: Initial observations to provide to the agent.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.
        """
        self.train_X = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        self.train_y = torch.tensor(
            data.y, dtype=self.dtype, device=self.device
        ).unsqueeze(-1)

        self.n_init = self.train_X.shape[0]  # Used for TuRBO restarts

        self.state = TuRBOState(
            dim=self.input_dim,
            batch_size=1,  # Will be updated on first ask
            best_value=self.train_y.max().item(),
        )

        self._fit_model()

    def tell(self, data: Dataset, seed: int) -> None:
        """Update the agent with new observations and refit the SingleTaskGP model.

        Args:
            data: Observations corresponding to the points requested in the previous
                `ask` call.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.

        Raises:
            RuntimeError: If the TuRBO state has not been initialized with
                `tell_initial_data`.
        """
        if self.state is None:
            msg = "TuRBO state has not been initialized. Call tell_initial_data first."
            raise RuntimeError(msg)

        new_X = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        new_y = torch.tensor(data.y, dtype=self.dtype, device=self.device).unsqueeze(-1)

        # Don't update the trust region if new_y is still from init buffer
        if self.model:
            self.state = update_state(self.state, new_y)

        self.train_X = torch.cat([self.train_X, new_X])
        self.train_y = torch.cat([self.train_y, new_y])

        if self.state.restart_triggered:
            # Reset state
            self.state = TuRBOState(
                dim=self.input_dim,
                batch_size=self.state.batch_size,
            )

            # Purge local data
            self.train_X = torch.empty(
                0, self.input_dim, dtype=self.dtype, device=self.device
            )
            self.train_y = torch.empty(0, 1, dtype=self.dtype, device=self.device)
            self.model = None

            # Fill up the restart buffer
            sobol = SobolEngine(self.input_dim, scramble=True, seed=seed)
            self._restart_buffer = sobol.draw(self.n_init).to(
                dtype=self.dtype, device=self.device
            )

        # No need to fit model when init buffer is still being unrolled
        elif self._restart_buffer.shape[0] == 0:
            self._fit_model()

    def ask(self, n: int, seed: int) -> Float[np.ndarray, "{n} D"]:
        """Select the next points to evaluate using TuRBO with Thompson Sampling.

        This method uses Thompson Sampling (MaxPosteriorSampling) to generate
        candidates within a trust region centered around the best point found so far.

        Args:
            n: Number of points to select (used for batch BO).
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.

        Returns:
            Points selected by the agent for evaluation.

        Raises:
            RuntimeError: If the TuRBO state has not been initialized with
                `tell_initial_data`.
        """
        if self.state is None:
            msg = "TuRBO state has not been initialized. Call tell_initial_data first."
            raise RuntimeError(msg)

        # Consume buffer: Feed pre-generated points sequentially
        if self._restart_buffer.shape[0] > 0:
            batch_n = min(n, self._restart_buffer.shape[0])
            samples = self._restart_buffer[:batch_n]
            self._restart_buffer = self._restart_buffer[batch_n:]
            return samples.cpu().numpy()

        # Update batch size in state if it has changed
        if self.state.batch_size != n:
            self.state.batch_size = n
            self.state.failure_tolerance = math.ceil(
                max([4.0 / n, float(self.input_dim) / n])
            )

        # Find the best point (center of trust region)
        best_idx = self.train_y.argmax()
        x_center = self.train_X[best_idx].clone()

        # Get trust region bounds (in [0,1]^d space)
        tr_lb, tr_ub = self._get_trust_region_bounds(x_center, self.state.length)

        # Use Sobol to generate perturbations
        if self.acqf_opt_strategy == "default":
            n_candidates = min(100 * self.input_dim, 5000)
        elif self.acqf_opt_strategy in {"no_raasp", "raasp"}:
            # For "no_raasp" and "raasp" we fix the number of candidates
            n_candidates = 5000
        sobol = SobolEngine(self.input_dim, scramble=True, seed=seed)
        sobol_samples = sobol.draw(n_candidates).to(
            dtype=self.dtype, device=self.device
        )
        sobol_samples = tr_lb + (tr_ub - tr_lb) * sobol_samples

        if self.acqf_opt_strategy in {"default", "raasp"}:
            # Candidates are formed by perturbing (a subset of dimensions of) the
            # center point within the trust region.
            prob_perturb = min(20.0 / self.input_dim, 1.0)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            mask = (
                torch.rand(
                    n_candidates,
                    self.input_dim,
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
                        self.input_dim,
                        size=(len(ind),),
                        device=self.device,
                        generator=generator,
                    ),
                ] = 1

            # Create candidate points from the center with perturbations
            X_cand = x_center.expand(n_candidates, self.input_dim).clone()
            X_cand[mask] = sobol_samples[mask]
        else:
            # For "no_raasp", candidates are sampled uniformly within the trust region
            X_cand = sobol_samples

        # Sample on the candidate points using Thompson Sampling
        with gpytorch.settings.max_cholesky_size(float("inf")):  # Always use Cholesky
            thompson_sampling = MaxPosteriorSampling(
                model=self.model, replacement=False
            )
            with torch.no_grad():  # We don't need gradients when using TS
                X_cand = X_cand.unsqueeze(0)
                candidates = thompson_sampling(X_cand, num_samples=n).squeeze(0)

        return candidates.detach().cpu().numpy()
