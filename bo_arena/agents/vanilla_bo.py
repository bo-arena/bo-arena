"""Vanilla BO agent."""

import numpy as np
import torch
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import gen_candidates_scipy
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from jaxtyping import Float

from bo_arena.agents.ask_tell_agent import AskTellAgent
from bo_arena.utils.dataset import Dataset


class VanillaBO(AskTellAgent):
    """Hvarfner's Vanilla BO.

    This agent implements the method described in "Vanilla Bayesian Optimization
    Performs Great in High Dimensions" by Hvarfner et al. (2024), which relies on an RBF
    kernel with a dimensionality-scaled prior (DSP) applied to the lengthscales.

    Reference:
        Hvarfner, C., Hellsten, E. O., & Nardi, L. (2024). Vanilla Bayesian Optimization
        Performs Great in High Dimensions. International Conference on Machine Learning
        (ICML).
    """

    def __init__(
        self,
        input_dim: int,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        **kwargs: object,  # ruff: ignore[unused-method-argument]
    ) -> None:
        """Initialize the VanillaBO agent.

        Args:
            input_dim: Dimensionality of the input space (assumed to be [0,1]^d).
            dtype: The torch dtype to use for the model and data.
            device: The torch device to use for the model and data.
            kwargs: Additional arguments.

        Raises:
            ValueError: If an unknown acqf_opt_strategy is given.
        """
        self.dtype = dtype
        self.device = device
        self.input_dim = input_dim
        # sample_around_best=True doubles effective raw_samples
        self.opt_kwargs = {
            "raw_samples": 2500,
            "num_restarts": 5,
            "retry_on_optimization_warning": False,
            "options": {
                "nonnegative": False,
                "sample_around_best": True,
                "sample_around_best_sigma": 0.1,
                "maxiter": 300,
                "batch_limit": 64,
            },
        }

        self.train_X = torch.empty(0, self.input_dim, dtype=dtype, device=device)
        self.train_y = torch.empty(0, 1, dtype=dtype, device=device)
        self.model = None

        # Bounds are always [0,1]^d since Bencher normalizes inputs
        self.bounds = torch.stack([
            torch.zeros(self.input_dim, dtype=self.dtype, device=self.device),
            torch.ones(self.input_dim, dtype=self.dtype, device=self.device),
        ])

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return "vanilla_bo"

    def _fit_model(self) -> None:
        """Fit a SingleTaskGP model (with DSP) using ExactMarginalLogLikelihood."""
        self.model = SingleTaskGP(
            self.train_X,
            self.train_y,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def tell_initial_data(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument]
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
        self._fit_model()

    def tell(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument]
        """Update the agent with new observations and refit the SingleTaskGP model.

        Args:
            data: Observations corresponding to the points requested in the previous
                `ask` call.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.
        """
        new_X = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        new_y = torch.tensor(data.y, dtype=self.dtype, device=self.device).unsqueeze(-1)

        self.train_X = torch.cat([self.train_X, new_X])
        self.train_y = torch.cat([self.train_y, new_y])
        self._fit_model()

    def ask(self, n: int, seed: int) -> Float[np.ndarray, "{n} D"]:  # ruff: ignore[unused-method-argument]
        """Select the next points to evaluate using qLogNEI.

        This method optimizes the q-batch Log-Noisy-Expected-Improvement acquisition
        function, matching the original code from Hvarfner et al. (2024).

        Args:
            n: Number of points to select (used for batch BO).
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.

        Returns:
            Points selected by the agent for evaluation.

        Raises:
            RuntimeError: If the model has not been initialized with
                `tell_initial_data`.
        """
        if self.model is None:
            msg = "Model has not been initialized. Call tell_initial_data first."
            raise RuntimeError(msg)

        # Hvarfner et al. use the MC-based qLogNEI for all batch sizes, including
        # for q=1.
        acq_func = qLogNoisyExpectedImprovement(
            model=self.model,
            X_baseline=self.train_X,
        )

        candidates, _ = optimize_acqf(
            acq_func,
            bounds=self.bounds,
            q=n,
            gen_candidates=gen_candidates_scipy,
            **self.opt_kwargs,
        )

        return candidates.detach().cpu().numpy()
