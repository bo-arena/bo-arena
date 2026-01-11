"""Arena for running BO experiments."""

import time
from typing import TYPE_CHECKING

import numpy as np

from bo_arena.utils.dataset import Dataset

if TYPE_CHECKING:
    from logging import Logger

    from wandb import Run

    from bo_arena.ask_tell_agent import AskTellAgent
    from bo_arena.benchmarks import Benchmark


class Arena:
    """Arena for running BO experiments.

    Args:
        benchmark: The benchmark to run.
        agent: The BO agent to use for optimization.
        seed: Random seed for reproducibility.
        n_initial_points: Number of initial points to sample.
        n_total_evaluations: Number of total evaluations.
        batch_size: Number of points to evaluate in each BO iteration.
        logger: (Optional) Logger for logging information.
        wandb_run: (Optional) WandB run to use for logging.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        agent: AskTellAgent,
        seed: int,
        n_initial_points: int | None,
        n_total_evaluations: int | None,
        batch_size: int,
        logger: Logger | None,
        wandb_run: Run | None,
    ) -> None:
        self.benchmark = benchmark
        self.agent = agent
        self.seed = seed

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

    def run(self) -> Dataset:
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
        """
        # Sample initial points from the benchmark's input space and evaluate them.
        initial_points = self.benchmark.metadata.input_space.sample(
            self.n_initial_points,
            self.seed,
        )
        initial_observations = self.benchmark.evaluate_points(initial_points)
        self.observed_data = self.observed_data + Dataset(
            initial_points, initial_observations
        )
        self.agent.tell_initial_data(self.observed_data)

        # Log initial points info.
        if self.logger is not None:
            self.logger.info(
                f"Initial points evaluated: {self.n_initial_points}, "
                f"Initial best observation: {float(np.min(initial_observations)):.4f}"
            )

        if self.wandb_run is not None:
            for iter in range(self.n_initial_points):
                observation_val = float(initial_observations[iter])
                best_val_so_far = float(np.min(initial_observations[: iter + 1]))
                is_best_observation = observation_val == best_val_so_far
                self.wandb_run.log({
                    "batch_index": 0,
                    "iteration": iter,
                    "observation": observation_val,
                    "best_observation": best_val_so_far,
                    "is_best_observation": is_best_observation,
                })

        # Run core BO loop.
        bo_loop_iters = (
            self.n_total_evaluations - self.n_initial_points
        ) // self.batch_size
        for batch_idx in range(bo_loop_iters):
            # Ask the agent for new points to evaluate.
            start = time.time()
            new_points = self.agent.ask(self.batch_size)
            ask_time = time.time() - start

            # Evaluate the benchmark at the new points.
            start = time.time()
            new_observations = self.benchmark.evaluate_points(new_points)
            eval_time = time.time() - start

            # Update the observed data with the new points and observations.
            new_points_dataset = Dataset(new_points, new_observations)
            self.observed_data = self.observed_data + new_points_dataset

            # Provide the new observations to the agent.
            start = time.time()
            self.agent.tell(new_points_dataset)
            tell_time = time.time() - start

            # Log iteration info (note, only logs info about the entire batch, not
            # individual points).
            if self.logger is not None:
                new_best_in_batch = np.min(new_observations) < np.min(
                    self.observed_data.y[: -self.batch_size]
                )
                self.logger.info(
                    f"BO loop iteration {batch_idx + 1}/{bo_loop_iters}: "
                    f"Total evaluations: {self.observed_data.n}, "
                    f"Ask time: {ask_time:.4f}s, "
                    f"Eval time: {eval_time:.4f}s, "
                    f"Tell time: {tell_time:.4f}s, "
                    f"Best observation: {float(np.min(self.observed_data.y)):.4f}"
                    f"New best observation in current batch: {new_best_in_batch}"
                )

            # Log detailed metrics to WandB for each point in the batch.
            if self.wandb_run is not None:
                for iter in range(
                    self.n_initial_points + batch_idx * self.batch_size,
                    self.n_initial_points + (batch_idx + 1) * self.batch_size,
                ):
                    observation_val = float(self.observed_data.y[iter])
                    best_val_so_far = float(np.min(self.observed_data.y[: iter + 1]))
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

        return self.observed_data
