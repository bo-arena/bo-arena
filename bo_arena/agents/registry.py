"""Registry for BO agents."""

from .bounce import Bounce
from .constrained_vanilla_bo import ConstrainedVanillaBO
from .ibnn import IBNN
from .linear import LinearBO
from .random import Random
from .saasbo import SAASBO
from .turbo import TuRBO
from .vanilla_bo import VanillaBO

AGENTS = {
    "bounce": Bounce,
    "constrained_vanilla_bo": ConstrainedVanillaBO,
    "ibnn": IBNN,
    "linear_bo": LinearBO,
    "random": Random,
    "saasbo": SAASBO,
    "turbo": TuRBO,
    "vanilla_bo": VanillaBO,
}
