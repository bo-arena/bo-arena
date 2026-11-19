"""Constrained Vanilla BO agent."""

from collections.abc import Callable

import numpy as np
import torch
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.generation import gen_candidates_scipy
from botorch.models import ModelListGP, SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import SumMarginalLogLikelihood
from jaxtyping import Float

from bo_arena.agents.ask_tell_agent import AskTellAgent
from bo_arena.problems.metadata import ProblemMetadata
from bo_arena.utils.dataset import Dataset


class ConstrainedVanillaBO(AskTellAgent):
    """Hvarfner's Vanilla BO extended to black-box constraints.

    The objective and each constraint are modelled by independent SingleTaskGPs
    (with DSP), combined into a `ModelListGP`. Candidates are selected with the
    constrained MC-based qLogNEI, where each constraint's feasibility bounds come from
    `problem_metadata.constraints`.

    Reference:
        Hvarfner, C., Hellsten, E. O., & Nardi, L. (2024). Vanilla Bayesian Optimization
        Performs Great in High Dimensions. International Conference on Machine Learning
        (ICML).
    """

    supports_constraints = True

    def __init__(
        self,
        problem_metadata: ProblemMetadata,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        **kwargs: object,  # ruff: ignore[unused-method-argument]
    ) -> None:
        """Initialize the ConstrainedVanillaBO agent.

        Args:
            problem_metadata: Metadata of the problem being optimized. See
                `AskTellAgent` for which fields agents should use.
            dtype: The torch dtype to use for the model and data.
            device: The torch device to use for the model and data.
            kwargs: Additional arguments.
        """
        self.dtype = dtype
        self.device = device
        self.input_dim = problem_metadata.input_dim
        self.constraint_specs = problem_metadata.constraints
        self.n_constraints = problem_metadata.n_constraints
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
        self.train_C = torch.empty(0, self.n_constraints, dtype=dtype, device=device)
        self.model = None

        # Bounds are always [0,1]^d since the Arena normalizes inputs
        self.bounds = torch.stack([
            torch.zeros(self.input_dim, dtype=self.dtype, device=self.device),
            torch.ones(self.input_dim, dtype=self.dtype, device=self.device),
        ])

        self.botorch_constraints: list[Callable[[torch.Tensor], torch.Tensor]] = []
        for i, spec in enumerate(self.constraint_specs, start=1):
            if spec.upper is not None:
                self.botorch_constraints.append(
                    lambda Z, i=i, u=spec.upper: Z[..., i] - u
                )
            if spec.lower is not None:
                self.botorch_constraints.append(
                    lambda Z, i=i, l=spec.lower: l - Z[..., i]
                )

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return "constrained_vanilla_bo"

    def _fit_model(self) -> None:
        """Fit a ModelListGP of SingleTaskGPs (with DSP) for objective and constraints.

        Output 0 models the objective, output i + 1 the i-th constraint. The sum MLL
        holds one ExactMarginalLogLikelihood per GP, and BoTorch fits each GP
        separately, exactly as in `VanillaBO`.
        """
        train_Y = torch.cat([self.train_y, self.train_C], dim=-1)
        self.model = ModelListGP(*[
            SingleTaskGP(self.train_X, train_Y[:, i : i + 1].contiguous())
            for i in range(train_Y.shape[-1])
        ])
        mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def tell_initial_data(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument]
        """Provide initial observations to the agent and fit the models.

        Args:
            data: Initial observations to provide to the agent.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.
        """
        self.train_X = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        self.train_y = torch.tensor(
            data.objectives, dtype=self.dtype, device=self.device
        )
        C = (
            np.stack([data.constraints[s.name] for s in self.constraint_specs], axis=-1)
            if self.constraint_specs
            else np.empty((data.n, 0))
        )
        self.train_C = torch.tensor(C, dtype=self.dtype, device=self.device)
        self._fit_model()

    def tell(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument]
        """Update the agent with new observations and refit the models.

        Args:
            data: Observations corresponding to the points requested in the previous
                `ask` call.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.
        """
        new_X = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        new_y = torch.tensor(data.objectives, dtype=self.dtype, device=self.device)
        C = (
            np.stack([data.constraints[s.name] for s in self.constraint_specs], axis=-1)
            if self.constraint_specs
            else np.empty((data.n, 0))
        )
        new_C = torch.tensor(C, dtype=self.dtype, device=self.device)

        self.train_X = torch.cat([self.train_X, new_X])
        self.train_y = torch.cat([self.train_y, new_y])
        self.train_C = torch.cat([self.train_C, new_C])
        self._fit_model()

    def ask(self, n: int, seed: int) -> Float[np.ndarray, "{n} D"]:  # ruff: ignore[unused-method-argument]
        """Select the next points to evaluate using constrained qLogNEI.

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

        acq_func = qLogNoisyExpectedImprovement(
            model=self.model,
            X_baseline=self.train_X,
            objective=GenericMCObjective(lambda Z, X: Z[..., 0]),  # ruff: ignore[unused-lambda-argument]
            constraints=self.botorch_constraints or None,
        )

        candidates, _ = optimize_acqf(
            acq_func,
            bounds=self.bounds,
            q=n,
            gen_candidates=gen_candidates_scipy,
            **self.opt_kwargs,
        )

        return candidates.detach().cpu().numpy()
