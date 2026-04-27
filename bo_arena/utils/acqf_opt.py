"""Acquisition function optimization strategy utilities."""

import numpy as np
import torch
from jaxtyping import Float
from torch.quasirandom import SobolEngine

from bo_arena.utils.miscellaneous import generate_seed


def get_log_ei_opt_kwargs(strategy: str) -> dict:
    """Return optimize_acqf keyword arguments for the given strategy name.

    Note that these settings are used by agents which use the logEI acquisition
    function (hence `nonnegative=False`).

    Args:
        strategy: One of "no_raasp", or "botorch_raasp".

    Returns:
        A dict of kwargs to pass to optimize_acqf.

    Raises:
        ValueError: If an unknown strategy name is given.
    """
    strategies: dict[str, dict] = {
        "no_raasp": {
            "raw_samples": 5000,
            "num_restarts": 5,
            "retry_on_optimization_warning": False,
            "options": {
                "nonnegative": False,
                "maxiter": 300,
                "batch_limit": 64,
            },
        },
        "botorch_raasp": {
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


def get_candidates(
    n_candidates: int,
    lb: Float[torch.Tensor, " D"],
    ub: Float[torch.Tensor, " D"],
    best_x: Float[torch.Tensor, " D"],
    seed: int,
    method: str,
    dtype: torch.dtype,
    device: torch.device | str,
) -> Float[torch.Tensor, "{n_candidates} D"]:
    """Generate a discrete candidate set inside ``[lb, ub]``.

    Two schemes are supported:

    * ``"sobol"``: scrambled Sobol samples drawn in ``[lb, ub]``.
    * ``"raasp"``: the perturbation scheme from Eriksson et al. (2019). Each
      candidate starts from ``best_x`` and, for each dimension independently
      with probability ``min(20/D, 1)``, is replaced by the corresponding
      coordinate of a Sobol sample drawn in ``[lb, ub]``. At least one
      dimension is guaranteed to be perturbed per candidate.

    Reference:
        Eriksson, D., et al. (2019). Scalable Global Optimization via Local Bayesian
        Optimization. Neural Information Processing Systems (NeurIPS).

    Args:
        n_candidates: Number of candidate points to generate.
        lb: Per-dimension lower bounds of the region to sample in.
        ub: Per-dimension upper bounds of the region to sample in.
        best_x: Best point found so far. Used by the ``"raasp"`` scheme. Ignored for
          ``"sobol"``.
        seed: Random seed for the Sobol engine and perturbation mask.
        method: One of ``"raasp"`` or ``"sobol"``.
        dtype: Torch dtype of the returned tensor.
        device: Torch device of the returned tensor.

    Returns:
        Candidate points.

    Raises:
        ValueError: If ``method`` is not one of ``"raasp"`` or ``"sobol"``.
    """
    if method not in {"raasp", "sobol"}:
        msg = f"Unknown candidate generation method: {method}."
        raise ValueError(msg)

    sobol_ss, mask_ss = np.random.SeedSequence(seed).spawn(2)
    sobol_seed = generate_seed(sobol_ss)
    mask_seed = generate_seed(mask_ss)

    input_dim = lb.shape[0]
    sobol = SobolEngine(input_dim, scramble=True, seed=sobol_seed)
    sobol_samples = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    sobol_samples = lb + (ub - lb) * sobol_samples

    if method == "sobol":
        return sobol_samples

    # RAASP: perturb a subset of dimensions of best_x toward sobol_samples.
    prob_perturb = min(20.0 / input_dim, 1.0)
    generator = torch.Generator(device=device).manual_seed(mask_seed)
    mask = (
        torch.rand(
            n_candidates,
            input_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        <= prob_perturb
    )
    # Ensure at least one dimension is perturbed for each candidate.
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    if len(ind) > 0:
        mask[
            ind,
            torch.randint(
                0,
                input_dim,
                size=(len(ind),),
                device=device,
                generator=generator,
            ),
        ] = 1

    X_cand = best_x.expand(n_candidates, input_dim).clone()
    X_cand[mask] = sobol_samples[mask]
    return X_cand
