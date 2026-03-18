"""Acquisition function optimization strategy configurations."""


def get_opt_kwargs(strategy: str) -> dict:
    """Return optimize_acqf keyword arguments for the given strategy name.

    Args:
        strategy: One of "no_raasp", or "raasp".

    Returns:
        A dict of kwargs to pass to optimize_acqf.

    Raises:
        ValueError: If an unknown strategy name is given.
    """
    strategies: dict[str, dict] = {
        "no_raasp": {
            "raw_samples": 5000,
            "num_restarts": 5,
        },
        "raasp": {
            # sample_around_best=True doubles effective raw_samples:
            # 2500 -> 5000, matching "no_raasp" above.
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
        },
    }

    if strategy not in strategies:
        msg = (
            f"Unknown acqf optimization strategy: {strategy!r}. "
            f"Choose from {list(strategies.keys())}."
        )
        raise ValueError(msg)

    return strategies[strategy]
