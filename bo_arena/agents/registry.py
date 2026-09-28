"""Registry for BO agents."""

from .bounce import Bounce
from .ibnn import IBNN
from .linear import LinearBO
from .random import Random
from .saasbo import SAASBO
from .turbo import TuRBO
from .vanilla_bo import VanillaBO

AGENTS = {
    "bounce": Bounce,
    "ibnn": IBNN,
    "linear_bo": LinearBO,
    "random": Random,
    "saasbo": SAASBO,
    "turbo": TuRBO,
    "vanilla_bo": VanillaBO,
}
