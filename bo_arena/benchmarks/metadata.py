"""Benchmarks metadata."""

from dataclasses import dataclass

import numpy as np
from bencherscaffold.protoclasses.bencher_pb2 import ValueType

from bo_arena.benchmarks.input_space import ContinuousInputSpace, InputSpace


# TODO(Thomas-Christie): Move all input info to input_space?
@dataclass(frozen=True)  # frozen=True makes them immutable
class BenchmarkMetadata:
    """Metadata that benchmarks must provide."""

    input_dim: int
    input_type: ValueType
    default_n_initial_points: int
    default_n_total_evaluations: int
    input_space: InputSpace


BENCHMARK_METADATA: dict[str, BenchmarkMetadata] = {
    "rover": BenchmarkMetadata(
        input_dim=60,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(60),
            upper_bounds=np.ones(60),
        ),
    ),  # TODO(Thomas-Christie): Check these values.
    "lasso-dna": BenchmarkMetadata(
        input_dim=180,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(180),
            upper_bounds=np.ones(180),
        ),
    ),
}
