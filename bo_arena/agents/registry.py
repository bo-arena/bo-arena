"""Registry for BO agents."""

from functools import partial

from .bounce import Bounce
from .ibnn import IBNN
from .linear import LinearBO
from .random import Random
from .saasbo import SAASBO
from .turbo import TuRBO
from .vanilla_bo import VanillaBO

AGENTS = {
    "bounce": Bounce,
    "bounce_log_ei": partial(Bounce, use_log_ei=True),
    "bounce_dsp": partial(Bounce, use_dsp=True),
    "bounce_log_ei_dsp": partial(Bounce, use_log_ei=True, use_dsp=True),
    "ibnn": IBNN,
    "linear_bo": LinearBO,
    "random": Random,
    "saasbo": SAASBO,
    "turbo": TuRBO,
    "vanilla_bo": VanillaBO,
}
