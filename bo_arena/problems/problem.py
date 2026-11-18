"""Base class for problems."""

from abc import abstractmethod
from collections.abc import Sequence

import numpy as np
from bencherscaffold.client import BencherClient
from bencherscaffold.protoclasses.bencher_pb2 import Constraint, ConstraintType, Value
from jaxtyping import Float, UInt32

from bo_arena.problems.metadata import PROBLEM_METADATA, ProblemMetadata
from bo_arena.utils.dataset import Dataset


class Problem:
    """Base class for problems.

    Note that problems should be written as maximization problems (following the same
    convention as in BoTorch).

    Args:
        name: Name of the problem.
        metadata: Problem metadata.
    """

    def __init__(self, name: str, metadata: ProblemMetadata) -> None:
        self.name = name
        self.metadata = metadata

    def __str__(self) -> str:
        """Return a string representation of the problem."""
        return self.name

    @abstractmethod
    def evaluate_points(
        self,
        points: Float[np.ndarray, "N D"],
        seeds: UInt32[np.ndarray, " N"] | None = None,
    ) -> Dataset:
        """Evaluate the problem at the given points.

        Args:
            points: Points to evaluate.
            seeds: (Optional) seeds, one per point, for seeding stochastic problems.

        Returns:
            Dataset containing the evaluated points and their objective values,
            plus one named constraint per entry in `metadata.constraints`.
        """
        msg = "This method should be implemented by subclasses."
        raise NotImplementedError(msg)


class BencherProblem(Problem):
    """Problem implementation using Bencher.

    Bencher returns constraint values alongside the objectives. Bencher must return
    exactly one constraint per entry in `metadata.constraints` (so none for an
    unconstrained problem), with matching names. The raw values are stored in
    `Dataset.constraints`, keyed by name. Bencher's equality constraints have no
    tolerance built in, so each must have both a lower and an upper bound in the
    metadata.

    Args:
        name: Name of the problem.
        metadata: Problem metadata. If None, will be looked up in PROBLEM_METADATA
          using the name.
        address: Address of the Bencher server. Can be a hostname (e.g. "127.0.0.1")
          or a Unix socket path (e.g. "unix:///path/to/bencher.sock").
        port: Port of the Bencher server. Ignored when address is a Unix socket.
    """

    def __init__(
        self,
        name: str,
        metadata: ProblemMetadata | None = None,
        address: str = "127.0.0.1",
        port: int = 50051,
    ) -> None:
        if metadata is None:
            metadata = PROBLEM_METADATA.get(name)
            if metadata is None:
                msg = f"Metadata not found for problem '{name}'."
                raise ValueError(msg)
        super().__init__(name, metadata)
        self.bencher_client = BencherClient(address=address, port=port)

    def _parse_constraints(self, constraints: Sequence[Constraint]) -> dict[str, float]:
        """Check Bencher's constraints against the metadata and extract their values.

        Args:
            constraints: Constraints returned by Bencher for a single point.

        Returns:
            Raw constraint values keyed by name.

        Raises:
            ValueError: If Bencher returns duplicate constraint names or names that
                differ from the metadata's, or if an equality constraint lacks a
                two-sided bound.
        """
        by_name = {constraint.name: constraint for constraint in constraints}
        if len(by_name) != len(constraints):
            names = [constraint.name for constraint in constraints]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            msg = (
                f"Bencher returned duplicate constraint names for problem "
                f"'{self.name}': {duplicates}."
            )
            raise ValueError(msg)

        expected = self.metadata.constraint_names
        if set(by_name) != set(expected):
            missing = sorted(set(expected) - set(by_name))
            unexpected = sorted(set(by_name) - set(expected))
            msg = (
                f"Bencher's constraint names for problem '{self.name}' do not match "
                f"its metadata: missing {missing}, unexpected {unexpected}."
            )
            raise ValueError(msg)

        for spec in self.metadata.constraints:
            if by_name[spec.name].type == ConstraintType.EQUALITY and (
                spec.lower is None or spec.upper is None
            ):
                msg = (
                    f"Constraint '{spec.name}' of problem '{self.name}' is an "
                    "equality constraint, so its metadata must set both a lower and "
                    f"an upper tolerance, got ({spec.lower}, {spec.upper})."
                )
                raise ValueError(msg)
        return {name: by_name[name].value for name in expected}

    def evaluate_points(
        self,
        points: Float[np.ndarray, "N D"],
        seeds: UInt32[np.ndarray, " N"] | None = None,
    ) -> Dataset:
        """Evaluate the problem at the given points.

        Bencher problems are minimization problems, so the objective values are
        negated to follow the maximization convention.

        Args:
            points: Points to evaluate.
            seeds: (Optional) seeds, one per point, for seeding stochastic problems.

        Returns:
            Dataset containing the evaluated points and their objective and
            constraint values.
        """
        objectives = []
        constraints: dict[str, list[float]] = {
            name: [] for name in self.metadata.constraint_names
        }
        for i, point in enumerate(points):
            bencher_point = [
                Value(type=self.metadata.input_type, value=float(d)) for d in point
            ]
            result = self.bencher_client.evaluate_point(
                benchmark_name=self.name,
                point=bencher_point,
                random_seed=int(seeds[i]) if seeds is not None else None,
            )
            # Turn minimization into maximization for BoTorch.
            objectives.append([-objective.value for objective in result.objectives])
            for name, value in self._parse_constraints(result.constraints).items():
                constraints[name].append(value)
        return Dataset(
            X=points,
            objectives=np.array(objectives),
            constraints={
                name: np.array(values) for name, values in constraints.items()
            },
        )
