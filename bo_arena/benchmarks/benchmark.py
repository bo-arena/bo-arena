"""Base class for benchmarks."""

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from bencherscaffold.client import BencherClient
from bencherscaffold.protoclasses.bencher_pb2 import Value

from bo_arena.benchmarks.metadata import BENCHMARK_METADATA, BenchmarkMetadata

if TYPE_CHECKING:
    from jaxtyping import Float


class Benchmark:
    """Base class for benchmarks.

    Args:
        name: Name of the benchmark.
        metadata: Benchmark metadata.
    """

    def __init__(self, name: str, metadata: BenchmarkMetadata) -> None:
        self.name = name
        self.metadata = metadata

    @abstractmethod
    def evaluate_points(
        self, points: Float[np.ndarray, "N D"]
    ) -> Float[np.ndarray, "N 1"]:
        """Evaluate the benchmark at the given points.

        Args:
            points: Points to evaluate of shape (N, D).

        Returns:
            Evaluations at the given points of shape (N, 1).
        """
        msg = "This method should be implemented by subclasses."
        raise NotImplementedError(msg)


class BencherBenchmark(Benchmark):
    """Benchmark implementation using Bencher."""

    def __init__(self, name: str, metadata: BenchmarkMetadata | None = None) -> None:
        if metadata is None:
            metadata = BENCHMARK_METADATA.get(name)
            if metadata is None:
                msg = f"Metadata not found for benchmark '{name}'."
                raise ValueError(msg)
        super().__init__(name, metadata)
        self.bencher_client = BencherClient()

    def evaluate_points(
        self, points: Float[np.ndarray, "N D"]
    ) -> Float[np.ndarray, "N 1"]:
        """Evaluate the benchmark at the given points.

        Args:
            points: Points to evaluate of shape (N, D).

        Returns:
            Evaluations at the given points of shape (N, 1).
        """
        evaluations = []
        for point in points:
            bencher_point = [
                Value(type=self.metadata.input_type, value=float(d)) for d in point
            ]  # TODO(Thomas-Christie): Will need to play with this.
            value = self.bencher_client.evaluate_point(
                benchmark_name=self.name,
                point=bencher_point,
            )
            evaluations.append(value)
        return np.array(evaluations).reshape(-1, 1)
