"""Registry for BO agents."""

from functools import partial

from .baxus import BAxUS
from .linear import LinearBO
from .random import Random
from .saasbo import SAASBO
from .turbo import TuRBO
from .vanilla_bo import VanillaBO, VanillaBOYu2026

AGENTS = {
    "baxus": BAxUS,
    "baxus_dsp": partial(BAxUS, use_dsp=True),
    "linear_bo": LinearBO,
    "random": Random,
    "saasbo": SAASBO,
    "turbo": TuRBO,
    "turbo_dsp": partial(TuRBO, use_dsp=True),
    "vanilla_bo": VanillaBO,
    "vanilla_bo_yu_2026": VanillaBOYu2026,
}
