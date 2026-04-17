"""Bounce (BO Using iNcreasingly high-dim Combinatorial/continuous Embeddings) agent."""

from typing import Any

import gpytorch
import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from jaxtyping import Float
from torch.quasirandom import SobolEngine

from bo_arena.agents.ask_tell_agent import AskTellAgent
from bo_arena.agents.baxus_bounce_utils import (
    BounceState,
    compute_adjustment_factor,
    embedding_matrix,
    increase_embedding_and_observations,
    update_bounce_state,
)
from bo_arena.utils.acqf_opt import get_opt_kwargs
from bo_arena.utils.dataset import Dataset


class Bounce(AskTellAgent):
    """Continuous-only Bounce agent (no binary/categorical inputs).

    This agent implements the continuous-only variant of Bounce as described in
    "Bounce: Reliable High-Dimensional Bayesian Optimization for Combinatorial and
    Mixed Spaces" by Papenmeier et al. (2023). Bounce starts optimization in a
    low-dimensional random subspace and adaptively expands the embedding until the
    full input dimensionality is reached, at which point the trust region resets.

    Reference:
        Papenmeier, L., Nardi, L., Poloczek, M. (2023). Bounce: Reliable
        High-Dimensional Bayesian Optimization for Combinatorial and Mixed
        Spaces. Neural Information Processing Systems (NeurIPS).
    """

    def __init__(
        self,
        input_dim: int,
        acqf_opt_strategy: str = "default",
        use_dsp: bool = False,  # noqa: FBT002
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the Bounce agent.

        Args:
            input_dim: Dimensionality of the input space (assumed [0,1]^d).
            acqf_opt_strategy: Strategy for optimizing the acquisition function. Can be
              "default", "no_raasp", or "botorch_raasp". Setting "default" preserves
              the original Bounce optimizer settings (raw_samples=512, num_restarts=10).
            use_dsp: If True, use BoTorch's default SingleTaskGP (Hvarfner's
              Dimensionality-Scaled Prior) instead of the Matern-2.5 + ARD + Gamma
              priors used in the original Bounce paper.
            dtype: The torch dtype to use for the model and data.
            device: The torch device to use for the model and data.
            kwargs: Additional arguments. 'n_total_evaluations' (int) is required.

        Raises:
            ValueError: If an unknown acqf_opt_strategy is given.

        """
        self.dtype = dtype
        self.device = device
        self.input_dim = input_dim
        self.use_dsp = use_dsp
        self.n_total_evaluations: int | None = kwargs.get("n_total_evaluations")
        if self.n_total_evaluations is None:
            msg = (
                "n_total_evaluations is required for Bounce. "
                "Pass it as a keyword argument to the constructor."
            )
            raise RuntimeError(msg)

        if acqf_opt_strategy == "default":
            self.opt_kwargs = {
                "raw_samples": 512,
                "num_restarts": 10,
                "options": {
                    "batch_limit": 64,
                },
            }
        elif acqf_opt_strategy in {"no_raasp", "botorch_raasp"}:
            self.opt_kwargs = get_opt_kwargs(acqf_opt_strategy)
        else:
            msg = f"Unknown acqf_opt_strategy: {acqf_opt_strategy!r}"
            raise ValueError(msg)

        self.train_X: torch.Tensor | None = None  # stored in [0, 1]^target_dim
        self.train_y = torch.empty(0, 1, dtype=dtype, device=device)
        self.model: SingleTaskGP | None = None
        self.state: BounceState | None = None
        self._n_evals: int = 0  # total evaluations including initial data
        self._restart_buffer = torch.empty(
            0, self.input_dim, dtype=dtype, device=device
        )

        # Projection matrix
        self.S: Float[torch.Tensor, "target_dim input_dim"] | None = None

        # Coordinate conventions used throughout this class:
        #   Arena / external   : [0, 1]^input_dim
        #   Full-dim internal  : [-1, 1]^input_dim  (self.bounds below)
        #   Subspace / train_X : [0, 1]^target_dim  (GP is trained in this space)
        self.bounds = torch.stack([
            -torch.ones(self.input_dim, dtype=self.dtype, device=self.device),
            torch.ones(self.input_dim, dtype=self.dtype, device=self.device),
        ])

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return "bounce_dsp" if self.use_dsp else "bounce"

    @property
    def default_n_initial_points(self) -> int:
        """Bounce always requires exactly 5 initial points."""
        return 5

    def _create_state(self, n_initial_points: int) -> BounceState:
        """Create a new BounceState."""
        return BounceState(
            input_dim=self.input_dim,
            n_initial_points=n_initial_points,
            eval_budget=self.n_total_evaluations,
        )

    def _fit_model(self) -> None:
        """Fit a SingleTaskGP model in the low-dimensional subspace.

        When use_dsp=False, uses Matern-2.5 with ARD and Gamma priors on train_X
        which is stored in [0,1]^target_dim. When use_dsp=True, uses BoTorch's
        default SingleTaskGP (Hvarfner's DSP). Falls back to Adam if standard
        fitting fails.
        """
        if self.use_dsp:
            self.model = SingleTaskGP(self.train_X, self.train_y)
        else:
            target_dim = self.train_X.shape[-1]
            likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.1, 0.05))
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=target_dim,
                    lengthscale_prior=GammaPrior(1.5, 0.1),
                ),
                outputscale_prior=GammaPrior(1.5, 0.5),
            )
            self.model = SingleTaskGP(
                self.train_X,
                self.train_y,
                covar_module=covar_module,
                likelihood=likelihood,
            )

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

        Computes bounds as ``center ± length * weights / 2`` clipped to [0, 1],
        where weights are ARD lengthscales normalized to unit geometric mean.

        Args:
            center: Center point of the trust region in [0,1]^target_dim.
            length: Length parameter of the trust region.

        Returns:
            A tuple of (tr_lb, tr_ub) tensors in [0,1]^target_dim.
        """
        if self.use_dsp:
            lengthscale = self.model.covar_module.lengthscale
        else:
            lengthscale = self.model.covar_module.base_kernel.lengthscale
        weights = lengthscale.squeeze().detach()
        if weights.ndim > 0:
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

        tr_lb = torch.clamp(center - length * weights / 2, 0.0, 1.0)
        tr_ub = torch.clamp(center + length * weights / 2, 0.0, 1.0)
        return tr_lb, tr_ub

    def generate_initial_points(self, n: int, seed: int) -> Float[np.ndarray, "N D"]:
        """Generate initial points in the low-dimensional subspace and project up.

        Initializes state and embedding matrix, generates Sobol points in the
        low-dimensional subspace, and projects them to [0,1]^d.

        Args:
            n: Number of initial points to generate.
            seed: Random seed for reproducibility.

        Returns:
            Initial points in [0,1]^d for the Arena to evaluate.
        """
        # Initialize state and embedding matrix early
        self.state = self._create_state(n_initial_points=n)
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
                "Bounce state has not been initialized. "
                "generate_initial_points must be called before tell_initial_data."
            )
            raise RuntimeError(msg)

        # Unnormalize from [0,1]^d (Arena) to [-1,1]^d (internal full-dim space)
        X_normalized = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        X_unnormalized = unnormalize(X=X_normalized, bounds=self.bounds)

        # Project X to low-dimensional subspace
        self.train_X = torch.linalg.lstsq(self.S.t(), X_unnormalized.t()).solution.t()

        # Scale from [-1,1]^target_dim to [0,1]^target_dim
        self.train_X = (self.train_X + 1.0) / 2.0
        self.train_y = torch.tensor(
            data.y, dtype=self.dtype, device=self.device
        ).unsqueeze(-1)

        self.n_init = self.train_X.shape[0]  # Used for Bounce restarts
        self._n_evals = self.train_X.shape[0]

        self._fit_model()

    def tell(self, data: Dataset, seed: int) -> None:
        """Update the agent with new observations and manage trust region geometry.

        Projects new data into the subspace, updates the trust region based on
        improvement, and handles embedding splits or full-dimensional restarts
        when the trust region terminates.

        Args:
            data: Observations corresponding to the points requested in the previous
                `ask` call.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.

        Raises:
            RuntimeError: If the Bounce state has not been initialized with
                `tell_initial_data`.
        """
        if self.state is None:
            msg = "Bounce state has not been initialized. Call tell_initial_data first."
            raise RuntimeError(msg)

        # Unnormalize from [0,1]^d (Arena) to [-1,1]^d (internal full-dim space)
        X_normalized = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        X_unnormalized = unnormalize(X=X_normalized, bounds=self.bounds)

        # Project X to low-dimensional subspace
        new_X = torch.linalg.lstsq(self.S.t(), X_unnormalized.t()).solution.t()

        # Scale from [-1,1]^target_dim to [0,1]^target_dim
        new_X = (new_X + 1.0) / 2.0
        new_y = torch.tensor(data.y, dtype=self.dtype, device=self.device).unsqueeze(-1)

        batch_size = new_X.shape[0]

        # Don't update the trust region if new data is still from the restart buffer
        if self.model is not None:
            y_best_prev = float(self.train_y.max().item())

            factor = compute_adjustment_factor(self.state, self._n_evals, batch_size)
            self.state = update_bounce_state(self.state, new_y, y_best_prev, factor)

            # Update per-tier budget tracking
            tr_dim = self.state.target_dim
            self.state.split_budgets[tr_dim] = (
                self.state.split_budgets.get(tr_dim, 1) - batch_size
            )

        self.train_X = torch.cat((self.train_X, new_X), dim=0)
        self.train_y = torch.cat([self.train_y, new_y])
        self._n_evals += batch_size

        if self.state.terminated:
            if self.state.target_dim < self.state.input_dim:
                # Increase target space: split each bin into new_bins_on_split+1 bins,
                # growing the embedding from target_dim to a higher dimensionality.
                generator = torch.Generator(device=self.device).manual_seed(seed)
                self.S, self.train_X = increase_embedding_and_observations(
                    self.S,
                    self.train_X,
                    self.state.new_bins_on_split,
                    generator=generator,
                )
                self.state.target_dim = len(self.S)

                # Allocate budget for the new tier if not already tracked
                if self.state.target_dim not in self.state.split_budgets:
                    self.state.split_budgets[self.state.target_dim] = (
                        self.state.split_budget_for(self.state.target_dim)
                    )

                # Partially reset trust region
                self.state.reset_trust_region()

            elif self.state.target_dim == self.state.input_dim:
                # Completely reset trust region and resample a fresh embedding
                self.state.reset_trust_region()
                self.state.split_budgets[self.state.target_dim] = (
                    self.state.split_budget_for(self.state.target_dim)
                )
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

        # No need to fit model when restart buffer is still being unrolled
        if self._restart_buffer.shape[0] == 0:
            self._fit_model()

    def ask(self, n: int, seed: int) -> Float[np.ndarray, "{n} D"]:
        """Select candidate points via (q)EI optimization in the trust region.

        Optimizes Expected Improvement or q-EI over the trust region in the
        low-dimensional subspace, then projects candidates back to the full space.

        Args:
            n: Number of points to select (used for batch BO).
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.

        Returns:
            Candidate points in [0,1]^d for evaluation.

        Raises:
            RuntimeError: If the Bounce state has not been initialized with
                `tell_initial_data`.
        """
        if self.state is None:
            msg = "Bounce state has not been initialized. Call tell_initial_data first."
            raise RuntimeError(msg)

        # Consume buffer: Feed pre-generated points sequentially
        if self._restart_buffer.shape[0] > 0:
            batch_n = min(n, self._restart_buffer.shape[0])
            samples = self._restart_buffer[:batch_n]
            self._restart_buffer = self._restart_buffer[batch_n:]
            return samples.cpu().numpy()

        # Find the best point (center of trust region) in [0,1]^target_dim
        best_idx = self.train_y.argmax()
        x_center = self.train_X[best_idx].clone()

        # Get trust region bounds (in [0,1]^target_dim)
        tr_lb, tr_ub = self._get_trust_region_bounds(
            x_center, self.state.length_continuous
        )
        bounds = torch.stack([tr_lb, tr_ub], dim=0)

        # Optimize (q)EI over the trust region
        best_f = float(self.train_y.max().item())
        with gpytorch.settings.max_cholesky_size(float("inf")):  # Always use Cholesky
            if n == 1:
                acqf = ExpectedImprovement(model=self.model, best_f=best_f)
                candidates, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=bounds,
                    q=1,
                    **self.opt_kwargs,
                )
            else:
                # Sequential greedy with X_pending
                sampler = SobolQMCNormalSampler(
                    sample_shape=torch.Size([1024]), seed=seed
                )
                cand_list: list[torch.Tensor] = []
                for _ in range(n):
                    x_pending = torch.cat(cand_list, dim=0) if cand_list else None
                    acqf = qExpectedImprovement(
                        model=self.model,
                        best_f=best_f,
                        sampler=sampler,
                        X_pending=x_pending,
                    )
                    x_new, _ = optimize_acqf(
                        acq_function=acqf,
                        bounds=bounds,
                        q=1,
                        **self.opt_kwargs,
                    )
                    cand_list.append(x_new)
                candidates = torch.cat(cand_list, dim=0)

        # Scale from [0,1]^target_dim back to [-1,1]^target_dim
        candidates = candidates * 2.0 - 1.0

        # Project X to full-dimensional input space, i.e. [-1,1]^d
        X_next = candidates @ self.S

        # BO-Arena assumes input space is [0,1]^d
        X_next_normalized = normalize(X=X_next, bounds=self.bounds)
        return X_next_normalized.detach().cpu().numpy()
