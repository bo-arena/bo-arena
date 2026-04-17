"""Registry for BO agents."""

from functools import partial

from .baxus import BAxUS
from .bounce import Bounce
from .linear import LinearBO
from .random import Random
from .saasbo import SAASBO
from .turbo import TuRBO, TuRBOLogEI
from .vanilla_bo import BestWindowVanillaBO, VanillaBO, VanillaBOYu2026, WindowVanillaBO

AGENTS = {
    "baxus": BAxUS,
    "baxus_dsp": partial(BAxUS, use_dsp=True),
    "bounce": Bounce,
    "bounce_dsp": partial(Bounce, use_dsp=True),
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
