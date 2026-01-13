"""Base class for benchmarks."""

from abc import abstractmethod

import numpy as np
from bencherscaffold.client import BencherClient
from bencherscaffold.protoclasses.bencher_pb2 import Value
from jaxtyping import Float

from bo_arena.benchmarks.metadata import BENCHMARK_METADATA, BenchmarkMetadata


class Benchmark:
    """Base class for benchmarks.

    Args:
        name: Name of the benchmark.
        metadata: Benchmark metadata.
        negate: Negate the underlying function.
    """

    def __init__(self, name: str, metadata: BenchmarkMetadata, negate: bool) -> None:
        self.name = name
        self.metadata = metadata
        self.negate = negate

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

    Bencher defines its benchmarks as minimization problems. Since BoTorch expects
    maximization problems, we negate all of Bencher's benchmarks by default (with
    negate=True).
    """

    def __init__(
        self,
        name: str,
        metadata: BenchmarkMetadata | None = None,
        *,
        negate: bool = True,
    ) -> None:
        if metadata is None:
            metadata = BENCHMARK_METADATA.get(name)
            if metadata is None:
                msg = f"Metadata not found for benchmark '{name}'."
                raise ValueError(msg)
        super().__init__(name, metadata, negate=negate)
        self.bencher_client = BencherClient()

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
            ]  # TODO(Thomas-Christie): Will need to play with this.
            value = self.bencher_client.evaluate_point(
                benchmark_name=self.name,
                point=bencher_point,
            )
            evaluations.append(value)
        evaluations = np.array(evaluations)
        if self.negate:
            evaluations = -evaluations
        return evaluations
