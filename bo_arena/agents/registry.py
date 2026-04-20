"""Registry for BO agents."""

from functools import partial

from .bounce import Bounce
from .linear import LinearBO
from .random import Random
from .saasbo import SAASBO
from .turbo import TuRBO, TuRBOLogEI
from .vanilla_bo import BestWindowVanillaBO, VanillaBO, VanillaBOYu2026, WindowVanillaBO

AGENTS = {
    "bounce": Bounce,
    "bounce_log_ei": partial(Bounce, use_log_ei=True),
    "bounce_dsp": partial(Bounce, use_dsp=True),
    "bounce_log_ei_dsp": partial(Bounce, use_log_ei=True, use_dsp=True),
    "linear_bo": LinearBO,
    "random": Random,
    "saasbo": SAASBO,
    "turbo": TuRBO,
    "turbo_dsp": partial(TuRBO, use_dsp=True),
    "turbo_log_ei": TuRBOLogEI,
    "turbo_log_ei_dsp": partial(TuRBOLogEI, use_dsp=True),
    "vanilla_bo": VanillaBO,
    "vanilla_bo_yu_2026": VanillaBOYu2026,
    "best_window_vanilla_bo": BestWindowVanillaBO,
    "window_vanilla_bo": WindowVanillaBO,
}
