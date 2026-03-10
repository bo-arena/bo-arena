"""Arena for running BO experiments."""

import time
from logging import Logger

import numpy as np
from wandb import Run

from bo_arena.agents import AskTellAgent
from bo_arena.benchmarks import Benchmark
from bo_arena.benchmarks.input_space import ContinuousInputSpace
from bo_arena.utils.dataset import Dataset
from bo_arena.utils.miscellaneous import (
    generate_seed,
    seed_everything,
    split_np,
)


class Arena:
    """Arena for running BO experiments.

    Note that benchmarks are assumed to be maximization problems, following the same
    convention as in BoTorch.

    Args:
        benchmark: The benchmark to run.
        agent: The BO agent to use for optimization.
        seed: Random seed for reproducibility.
        batch_size: Number of points to evaluate in each BO iteration.
        n_initial_points: (Optional) number of initial points to sample - if None, uses
            the benchmark's default.
        n_total_evaluations: (Optional) number of total evaluations - if None, uses the
            benchmark's default.
        logger: (Optional) logger for logging information.
        wandb_run: (Optional) WandB run to use for logging.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        agent: AskTellAgent,
        seed: int,
        batch_size: int,
        n_initial_points: int | None = None,
        n_total_evaluations: int | None = None,
        logger: Logger | None = None,
        wandb_run: Run | None = None,
    ) -> None:
        self.benchmark = benchmark
        self.agent = agent
        self._key = np.random.SeedSequence(seed)
        self._has_run = False

        if n_initial_points is None:
            self.n_initial_points = benchmark.metadata.default_n_initial_points
        else:
            self.n_initial_points = n_initial_points

        if n_total_evaluations is None:
            self.n_total_evaluations = benchmark.metadata.default_n_total_evaluations
        else:
            self.n_total_evaluations = n_total_evaluations

        self.batch_size = batch_size
        self.logger = logger
        self.wandb_run = wandb_run
        self.observed_data = Dataset(
            np.empty((0, self.benchmark.metadata.input_dim)), np.empty(0)
        )

    def _derive_next_seed(self) -> int:
        """Split the key and return a derived integer seed."""
        self._key, subkey = split_np(self._key, n=2)
        return generate_seed(subkey)

    def run(self) -> Dataset:  # noqa: PLR0915
        """Run the main BO loop.

        Runs the main Bayesian optimization loop, which consists of:
        1. Sampling initial points from the benchmark's input space and evaluating them.
        2. Iteratively asking the agent for new points to evaluate, evaluating them on
           the benchmark, and providing the observations back to the agent.

        If a logger is provided, logs information about each iteration, including
        the number of evaluations, times taken for each step, and the best observation
        so far. If a WandB run is provided, logs detailed metrics for each evaluation.

        Returns:
            The dataset of all observed points and their corresponding observations.

        Raises:
            RuntimeError: If the method is called more than once on the same instance.
                This is to prevent non-deterministic behavior due to the internal state
                of the Arena (e.g. seed counter, agent) being modified across runs.
        """
        if self._has_run:
            msg = (
                "Arena.run() has already been called. Create a new Arena instance for "
                "each run."
            )
            raise RuntimeError(msg)
        self._has_run = True

        # Sample initial points from the benchmark's input space and evaluate them.
        # Agents may override generate_initial_points to provide custom initialization.
        seed = self._derive_next_seed()
        seed_everything(seed)
        custom_points = self.agent.generate_initial_points(self.n_initial_points, seed)
        if custom_points is not None:
            initial_points = custom_points
        else:
            initial_points = self.benchmark.metadata.input_space.sample(
                self.n_initial_points,
                seed,
            )
        initial_points = initial_points.astype(np.float32)
        input_space = self.benchmark.metadata.input_space
        if isinstance(input_space, ContinuousInputSpace):
            initial_points = np.clip(
                initial_points,
                input_space.lower_bounds.astype(np.float32),
                input_space.upper_bounds.astype(np.float32),
            )
        initial_observations = self.benchmark.evaluate_points(initial_points).astype(
            np.float32
        )
        self.observed_data = self.observed_data + Dataset(
            initial_points, initial_observations
        )
        seed = self._derive_next_seed()
        seed_everything(seed)
        self.agent.tell_initial_data(self.observed_data, seed)

        # Log initial points info.
        if self.logger is not None:
            self.logger.info(
                f"Initial points evaluated: {self.n_initial_points}\n"
                f"Initial best observation: {float(np.max(initial_observations)):.4f}"
            )

        if self.wandb_run is not None:
            for iter in range(self.n_initial_points):
                observation_val = float(initial_observations[iter])
                best_val_so_far = float(np.max(initial_observations[: iter + 1]))
                is_best_observation = observation_val == best_val_so_far
                self.wandb_run.log({
                    "batch_index": 0,
                    "iteration": iter,
                    "observation": observation_val,
                    "best_observation": best_val_so_far,
                    "is_best_observation": is_best_observation,
                })

        # Run core BO loop. Uses a while loop rather than a fixed iteration count
        # because agents may return fewer points than requested (e.g. during
        # initialization phases).
        batch_idx = 0
        while self.observed_data.n < self.n_total_evaluations:
            remaining = self.n_total_evaluations - self.observed_data.n
            n_ask = min(self.batch_size, remaining)

            # Ask the agent for new points to evaluate.
            start = time.time()
            seed = self._derive_next_seed()
            seed_everything(seed)
            new_points = self.agent.ask(n_ask, seed)
            ask_time = time.time() - start
            new_points = new_points.astype(np.float32)
            if isinstance(input_space, ContinuousInputSpace):
                new_points = np.clip(
                    new_points,
                    input_space.lower_bounds.astype(np.float32),
                    input_space.upper_bounds.astype(np.float32),
                )

            # Evaluate the benchmark at the new points.
            start = time.time()
            new_observations = self.benchmark.evaluate_points(new_points).astype(
                np.float32
            )
            eval_time = time.time() - start

            # Update the observed data with the new points and observations.
            new_points_dataset = Dataset(new_points, new_observations)
            self.observed_data = self.observed_data + new_points_dataset

            # Provide the new observations to the agent.
            start = time.time()
            seed = self._derive_next_seed()
            seed_everything(seed)
            self.agent.tell(new_points_dataset, seed)
            tell_time = time.time() - start

            # Log iteration info (note, only logs info about the entire batch, not
            # individual points).
            actual_batch_size = len(new_points)
            if self.logger is not None:
                new_best_in_batch = np.max(new_observations) > np.max(
                    self.observed_data.y[:-actual_batch_size]
                )
                self.logger.info(
                    f"BO loop iteration {batch_idx + 1}:\n"
                    f"  Total evaluations: {self.observed_data.n}\n"
                    f"  Ask time: {ask_time:.4f}s\n"
                    f"  Eval time: {eval_time:.4f}s\n"
                    f"  Tell time: {tell_time:.4f}s\n"
                    f"  Best observation: {float(np.max(self.observed_data.y)):.4f}\n"
                    f"  New best observation in current batch: {new_best_in_batch}"
                )

            # Log detailed metrics to WandB for each point in the batch.
            if self.wandb_run is not None:
                batch_start = self.observed_data.n - actual_batch_size
                for iter in range(batch_start, self.observed_data.n):
                    observation_val = float(self.observed_data.y[iter])
                    best_val_so_far = float(np.max(self.observed_data.y[: iter + 1]))
                    is_best_observation = observation_val == best_val_so_far
                    self.wandb_run.log({
                        "batch_index": batch_idx + 1,
                        "iteration": iter,
                        "observation": observation_val,
                        "best_observation": best_val_so_far,
                        "is_best_observation": is_best_observation,
                        "ask_time": ask_time,
                        "eval_time": eval_time,
                        "tell_time": tell_time,
                    })

            batch_idx += 1

        return self.observed_data
