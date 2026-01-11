"""Base class for ask-tell agents."""

from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from jaxtyping import Float

    from bo_arena.utils.dataset import Dataset


class AskTellAgent:
    """Base class for ask-tell agents.

    Ask-tell agents must implement three main methods:
    - ask: called to obtain new points to evaluate.
    - tell: called to provide observations for previously requested points.
    - tell_initial_data: called to provide initial observations to the agent.

    In addition, a string representation method (__str__) is required, which should
    include useful information about the agent for experiment management.
    # TODO(Thomas-Christie): think about this once we have some agents.

    These methods are used in the Bayesian optimization loop to iteratively
    select points to evaluate and update the agent's internal model with new
    observations.
    """

    def __init__(
        self,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the agent."""
        raise NotImplementedError

    @abstractmethod
    def ask(self, n: int, seed: int) -> Float[np.ndarray, "{n} D"]:
        """Return points to evaluate next.

        Args:
            n: Number of points to select (used for batch BO).
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.

        Returns:
            Points selected by the agent for evaluation.
        """
        raise NotImplementedError

    @abstractmethod
    def tell(self, data: Dataset, seed: int) -> None:
        """Provide observations for previously requested points.

        Args:
            data: Observations corresponding to the points requested
                in the previous `ask` call.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.
        """
        raise NotImplementedError

    @abstractmethod
    def tell_initial_data(self, data: Dataset, seed: int) -> None:
        """Provide initial observations to the agent.

        This method is used to provide the agent with an initial dataset
        before any `ask` calls are made. It is typically called once at the
        start of the optimization process.

        Args:
            data: Initial observations to provide to the agent.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.
        """
        raise NotImplementedError
