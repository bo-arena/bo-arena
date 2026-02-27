"""Registry for BO agents."""

from .random import Random
from .saasbo import SAASBO
from .turbo import TuRBO
from .vanilla_bo import VanillaBO

AGENTS = {
    "random": Random,
    "saasbo": SAASBO,
    "turbo": TuRBO,
    "vanilla_bo": VanillaBO,
}
