"""Base class for benchmarks."""

from abc import abstractmethod

import numpy as np
from bencherscaffold.client import BencherClient
from bencherscaffold.protoclasses.bencher_pb2 import Value
from jaxtyping import Float

from bo_arena.benchmarks.metadata import BENCHMARK_METADATA, BenchmarkMetadata


class Benchmark:
    """Base class for benchmarks.

    Note that benchmarks should be written as maximization problems (following the same
    convention as in BoTorch).

    Args:
        name: Name of the benchmark.
        metadata: Benchmark metadata.
    """

    def __init__(self, name: str, metadata: BenchmarkMetadata) -> None:
        self.name = name
        self.metadata = metadata

    def __str__(self) -> str:
        """Return a string representation of the benchmark."""
        return self.name

    @abstractmethod
    def evaluate_points(
        self, points: Float[np.ndarray, "N D"]
    ) -> Float[np.ndarray, " N"]:
        """Evaluate the benchmark at the given points.

        Args:
            points: Points to evaluate.

        Returns:
            Evaluations at the given points.
        """
        msg = "This method should be implemented by subclasses."
        raise NotImplementedError(msg)


class BencherBenchmark(Benchmark):
    """Benchmark implementation using Bencher.

    Args:
        name: Name of the benchmark.
        metadata: Benchmark metadata. If None, will be looked up in BENCHMARK_METADATA
          using the name.
        hostname: Hostname of the Bencher server.
        port: Port of the Bencher server.
    """

    def __init__(
        self,
        name: str,
        metadata: BenchmarkMetadata | None = None,
        hostname: str = "127.0.0.1",
        port: int = 50051,
    ) -> None:
        if metadata is None:
            metadata = BENCHMARK_METADATA.get(name)
            if metadata is None:
                msg = f"Metadata not found for benchmark '{name}'."
                raise ValueError(msg)
        super().__init__(name, metadata)
        self.bencher_client = BencherClient(hostname=hostname, port=port)

    def evaluate_points(
        self, points: Float[np.ndarray, "N D"]
    ) -> Float[np.ndarray, " N"]:
        """Evaluate the benchmark at the given points.

        Args:
            points: Points to evaluate.

        Returns:
            Evaluations at the given points.
        """
        evaluations = []
        for point in points:
            bencher_point = [
                Value(type=self.metadata.input_type, value=float(d)) for d in point
            ]
            value = self.bencher_client.evaluate_point(
                benchmark_name=self.name,
                point=bencher_point,
            )
            evaluations.append(value)
        return -np.array(evaluations)  # turn minimization into maximization for BoTorch
