"""Acquisition function optimization strategy configurations."""


def get_opt_kwargs(strategy: str) -> dict | None:
    """Return optimize_acqf keyword arguments for the given strategy name.

    Args:
        strategy: One of "default", "no_raasp", or "raasp".

    Returns:
        A dict of kwargs to pass to optimize_acqf, or None for the agent default.

    Raises:
        ValueError: If an unknown strategy name is given.
    """
    strategies: dict[str, dict | None] = {
        "default": None,
        "no_raasp": {
            "num_restarts": 5,
            "raw_samples": 5000,
        },
        "raasp": {
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
        msg = f"Unknown acqf optimization strategy: {strategy!r}. Choose from {list(strategies.keys())}."  # noqa: E501
        raise ValueError(msg)

    return strategies[strategy]
