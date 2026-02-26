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
    input_type: int  # protobuf ValueType enum (not a PEP-compliant type)
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
    ),
    "mopta08": BenchmarkMetadata(
        input_dim=124,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(124),
            upper_bounds=np.ones(124),
        ),
    ),
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
    "svm": BenchmarkMetadata(
        input_dim=388,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(388),
            upper_bounds=np.ones(388),
        ),
    ),
    "mujoco-ant": BenchmarkMetadata(
        input_dim=888,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(888),
            upper_bounds=np.ones(888),
        ),
    ),
    "mujoco-humanoid": BenchmarkMetadata(
        input_dim=6392,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(6392),
            upper_bounds=np.ones(6392),
        ),
    ),
}


SYNTHETIC_BENCHMARK_METADATA: dict[str, BenchmarkMetadata] = {
    "hartmann6d_embed_dim_100": BenchmarkMetadata(
        input_dim=100,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=20,
        default_n_total_evaluations=200,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(100),
            upper_bounds=np.ones(100),
        ),
    )
}
