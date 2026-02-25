"""Synthetic benchmarks."""

import numpy as np
from jaxtyping import Float

from bo_arena.benchmarks import Benchmark
from bo_arena.benchmarks.metadata import SYNTHETIC_BENCHMARK_METADATA, BenchmarkMetadata

# Standard Hartmann 6D constants: https://www.sfu.ca/~ssurjano/hart6.html
_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_A = np.array([
    [10, 3, 17, 3.5, 1.7, 8],
    [0.05, 10, 17, 0.1, 8, 14],
    [3, 3.5, 1.7, 10, 17, 8],
    [17, 8, 0.05, 10, 0.1, 14],
])
_P = 1e-4 * np.array([
    [1312, 1696, 5569, 124, 8283, 5886],
    [2329, 4135, 8307, 3736, 1004, 9991],
    [2348, 1451, 3522, 2883, 3047, 6650],
    [4047, 8828, 8732, 5743, 1091, 381],
])


def _hartmann6d(x: Float[np.ndarray, "N 6"]) -> Float[np.ndarray, " N"]:
    """Evaluate the Hartmann 6D function (negated for maximisation).

    The standard Hartmann 6D is a minimisation problem with global minimum
    approximately -3.3224 at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652,
    0.6573). We negate it so that the Arena's maximisation convention is followed.

    Args:
        x: Input points of shape (N, 6) in [0, 1]^6.

    Returns:
        Function values of shape (N,).
    """
    inner = np.sum(
        _A[np.newaxis, :, :] * (x[:, np.newaxis, :] - _P[np.newaxis, :, :]) ** 2, axis=2
    )
    return np.sum(_ALPHA * np.exp(-inner), axis=1)


class Hartmann6D(Benchmark):
    """Hartmann 6D function (optionally) embedded in a higher-dimensional space.

    The first 6 dimensions are active; the remaining dimensions are ignored.
    The function is negated relative to the standard form so that it is a
    maximisation problem (following the Arena convention).

    Args:
        embed_dim: Total input dimensionality (must be >= 6).

    Raises:
        ValueError: If embed_dim is less than 6 or if metadata is not found for the
        benchmark.
    """

    def __init__(
        self, embed_dim: int = 6, metadata: BenchmarkMetadata | None = None
    ) -> None:
        if embed_dim < 6:
            msg = f"embed_dim must be >= 6, got {embed_dim}."
            raise ValueError(msg)

        name = f"hartmann6d_embed_dim_{embed_dim}"
        if metadata is None:
            metadata = SYNTHETIC_BENCHMARK_METADATA.get(name)
            if metadata is None:
                msg = f"Metadata not found for benchmark '{name}'."
                raise ValueError(msg)

        super().__init__(name=name, metadata=metadata)

    def evaluate_points(  # noqa: PLR6301
        self, points: Float[np.ndarray, "N D"]
    ) -> Float[np.ndarray, " N"]:
        """Evaluate the Hartmann 6D function at the given points.

        Args:
            points: Points to evaluate.

        Returns:
            Function values.
        """
        return _hartmann6d(points[:, :6])
