"""A collection of utility functions for processing results from experiments."""

import numpy as np
from jaxtyping import Float


def get_cum_max_y_stats(
    ys: Float[np.ndarray, "n_runs n_evals"],
) -> tuple[Float[np.ndarray, " n_evals"], Float[np.ndarray, " n_evals"]]:
    """Get the cumulative maximum y value statistics across multiple runs.

    Args:
        ys: A 2D array of shape (n_runs, n_evals) containing the y values observed
            during each run.

    Returns:
        A tuple containing two 1D arrays of shape (n_evals,):
            - The mean cumulative max y values across runs.
            - The standard error of the mean of the cumulative max y values across runs.
    """
    cum_max_ys = np.maximum.accumulate(ys, axis=1)
    mean_cum_max_ys = np.mean(cum_max_ys, axis=0)
    sem_cum_max_ys = np.std(cum_max_ys, axis=0) / np.sqrt(cum_max_ys.shape[0])
    return mean_cum_max_ys, sem_cum_max_ys
