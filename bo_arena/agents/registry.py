"""Registry for BO agents."""

from .baxus import BAxUS
from .random import Random
from .saasbo import SAASBO
from .turbo import TuRBO
from .vanilla_bo import VanillaBO

AGENTS = {
    "baxus": BAxUS,
    "random": Random,
    "saasbo": SAASBO,
    "turbo": TuRBO,
    "vanilla_bo": VanillaBO,
}
