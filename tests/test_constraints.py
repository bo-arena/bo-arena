"""Tests for constraint metadata, feasibility, and constrained Arena runs."""

import dataclasses

import numpy as np
import pytest
from bencherscaffold.protoclasses.bencher_pb2 import (
    Constraint,
    ConstraintType,
    EvaluationResult,
    ObjectiveValue,
    ValueType,
)
from jaxtyping import Float, UInt32

from bo_arena.agents.random import Random
from bo_arena.arena import Arena
from bo_arena.problems import BencherProblem, Problem
from bo_arena.problems.input_space import ContinuousInputSpace
from bo_arena.problems.metadata import ConstraintSpec, ProblemMetadata
from bo_arena.utils.constraints import is_feasible, running_best_feasible
from bo_arena.utils.dataset import Dataset

D = 3


def _metadata(
    constraint_bounds: tuple[tuple[float | None, float | None], ...] = (),
) -> ProblemMetadata:
    """Build metadata with constraints named c0, c1, ... from (lower, upper) pairs."""
    return ProblemMetadata(
        input_dim=D,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=4,
        default_n_total_evaluations=10,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(D), upper_bounds=np.ones(D)
        ),
        constraints=tuple(
            ConstraintSpec(f"c{i}", lower, upper)
            for i, (lower, upper) in enumerate(constraint_bounds)
        ),
    )


class _ConstrainedToy(Problem):
    """Maximise sum(x) subject to x_0 <= 0.5 and x_1 == 0.5 (up to 0.25).

    The constraints are named c0 and c1. `returned_names` controls which names the
    problem returns, and in what order, to test the Arena's validation.
    """

    def __init__(self, returned_names: tuple[str, ...] = ("c0", "c1")) -> None:
        super().__init__("toy", _metadata(((None, 0.5), (-0.25, 0.25))))
        self.returned_names = returned_names

    def evaluate_points(
        self,
        points: Float[np.ndarray, "N D"],
        seeds: UInt32[np.ndarray, " N"] | None = None,  # ruff: ignore[unused-method-argument]
    ) -> Dataset:
        values = {"c0": points[:, 0], "c1": points[:, 1] - 0.5}
        return Dataset(
            X=points,
            objectives=points.sum(axis=1, keepdims=True),
            constraints={name: values[name] for name in self.returned_names},
        )


class _RecordingAgent(Random):
    """Random agent that supports constraints and records the data it is told."""

    supports_constraints = True

    def __init__(self, problem_metadata: ProblemMetadata, **kwargs: object) -> None:
        super().__init__(problem_metadata, **kwargs)
        self.told: list[Dataset] = []

    def tell(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument]
        self.told.append(data)

    def tell_initial_data(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument]
        self.told.append(data)


@pytest.mark.parametrize(("lower", "upper"), [(None, None), (1.0, 0.0)])
def test_invalid_constraint_spec_raises(
    lower: float | None, upper: float | None
) -> None:
    """Specs with no bound set, or lower > upper, should be rejected."""
    with pytest.raises(ValueError, match="Constraint 'g'"):
        ConstraintSpec("g", lower=lower, upper=upper)


def test_metadata_constraint_properties() -> None:
    """Metadata should expose the constraint count and names."""
    assert _metadata().n_constraints == 0
    metadata = _metadata(((None, 0.0), (-1e-3, 1e-3)))
    assert metadata.n_constraints == 2
    assert metadata.constraint_names == ("c0", "c1")


def test_duplicate_constraint_names_raise() -> None:
    """Constraint names must be unique within a problem."""
    with pytest.raises(ValueError, match="unique"):
        ProblemMetadata(
            input_dim=D,
            input_type=ValueType.CONTINUOUS,
            default_n_initial_points=4,
            default_n_total_evaluations=10,
            input_space=ContinuousInputSpace(
                lower_bounds=np.zeros(D), upper_bounds=np.ones(D)
            ),
            constraints=(
                ConstraintSpec("g", upper=0.0),
                ConstraintSpec("g", upper=1.0),
            ),
        )


def _constrained_dataset(constraints: dict[str, list[float]]) -> Dataset:
    n = len(next(iter(constraints.values()))) if constraints else 3
    return Dataset(
        X=np.zeros((n, D)),
        objectives=np.zeros((n, 1)),
        constraints={name: np.array(values) for name, values in constraints.items()},
    )


def test_is_feasible() -> None:
    """Feasibility should respect one-sided, two-sided and missing bounds."""
    specs = (
        ConstraintSpec("a", upper=0.0),
        ConstraintSpec("b", lower=1.0),
        ConstraintSpec("c", lower=-0.1, upper=0.1),
    )
    data = _constrained_dataset({
        # feasible, feasible on the boundaries, violates a, violates b,
        # violates the equality tolerance on c, NaN is infeasible
        "a": [-1.0, 0.0, 0.5, -1.0, -1.0, np.nan],
        "b": [2.0, 1.0, 2.0, 0.5, 2.0, 2.0],
        "c": [0.0, 0.1, 0.0, 0.0, 0.2, 0.0],
    })
    np.testing.assert_array_equal(
        is_feasible(data, specs), [True, True, False, False, False, False]
    )


def test_is_feasible_unconstrained() -> None:
    """With no constraints, every observation should be feasible."""
    assert is_feasible(_constrained_dataset({}), ()).all()


def test_is_feasible_mismatched_names_raises() -> None:
    """The dataset's constraint names must match the specs' names."""
    data = _constrained_dataset({"a": [0.0]})
    with pytest.raises(ValueError, match="constraint specs"):
        is_feasible(data, (ConstraintSpec("b", upper=0.0),))


def test_running_best_feasible() -> None:
    """The running best should ignore infeasible points and be NaN until feasible."""
    observations = np.array([5.0, 1.0, 3.0, 9.0, 2.0])
    feasible = np.array([False, True, True, False, True])
    np.testing.assert_array_equal(
        running_best_feasible(observations, feasible), [np.nan, 1.0, 3.0, 3.0, 3.0]
    )


def test_running_best_feasible_ignores_nan_objectives() -> None:
    """A feasible NaN objective should not poison the running best."""
    observations = np.array([np.nan, 1.0, np.nan, 3.0, 2.0])
    feasible = np.array([True, True, True, True, True])
    np.testing.assert_array_equal(
        running_best_feasible(observations, feasible), [np.nan, 1.0, 1.0, 3.0, 3.0]
    )


def test_unconstrained_agent_rejected_on_constrained_problem() -> None:
    """Agents without constraint support are rejected on constrained problems."""
    problem = _ConstrainedToy()
    with pytest.raises(ValueError, match="does not support constraints"):
        Arena(
            problem=problem,
            agent=Random(problem_metadata=problem.metadata),
            seed=0,
            batch_size=2,
        )


def test_constraints_reach_agent() -> None:
    """Constraint values should flow from the problem to the agent, unscaled."""
    problem = _ConstrainedToy()
    agent = _RecordingAgent(problem_metadata=problem.metadata)
    result = Arena(problem=problem, agent=agent, seed=0, batch_size=2).run()
    told = agent.told[0]
    for data in agent.told[1:]:
        told += data
    assert tuple(told.constraints) == ("c0", "c1")
    for name in ("c0", "c1"):
        np.testing.assert_array_equal(
            told.constraints[name], result.observed_data.constraints[name]
        )
    assert agent.problem_metadata.constraints == problem.metadata.constraints


class _EmptyBatchAgent(_RecordingAgent):
    """Agent that returns no points from `ask`."""

    def ask(self, n: int, seed: int) -> Float[np.ndarray, "N D"]:  # ruff: ignore[unused-method-argument]
        return np.empty((0, self.input_dim))


def test_empty_batch_raises() -> None:
    """An agent returning no points should fail loudly rather than loop forever."""
    problem = _ConstrainedToy()
    arena = Arena(
        problem=problem,
        agent=_EmptyBatchAgent(problem_metadata=problem.metadata),
        seed=0,
        batch_size=2,
    )
    with pytest.raises(RuntimeError, match="returned no points"):
        arena.run()


def test_missing_constraint_raises() -> None:
    """A problem not returning all of its metadata's constraints fails."""
    problem = _ConstrainedToy(returned_names=("c0",))
    arena = Arena(
        problem=problem,
        agent=_RecordingAgent(problem_metadata=problem.metadata),
        seed=0,
        batch_size=2,
    )
    with pytest.raises(ValueError, match=r"missing \['c1'\], unexpected \[\]"):
        arena.run()


class _FakeBencherClient:
    """Stand-in for BencherClient that returns a fixed EvaluationResult."""

    def __init__(self, constraints: list[Constraint]) -> None:
        self.constraints = constraints

    def evaluate_point(self, **kwargs: object) -> EvaluationResult:  # ruff: ignore[unused-method-argument]
        return EvaluationResult(
            objectives=[ObjectiveValue(name="f", value=2.0)],
            constraints=self.constraints,
        )


def _bencher_problem(
    specs: tuple[ConstraintSpec, ...], constraints: list[Constraint]
) -> BencherProblem:
    metadata = dataclasses.replace(_metadata(), constraints=specs)
    problem = BencherProblem(name="fake", metadata=metadata)
    problem.bencher_client = _FakeBencherClient(constraints)
    return problem


_INEQ = Constraint(name="g", type=ConstraintType.INEQUALITY, value=-1.5)
_EQ = Constraint(name="h", type=ConstraintType.EQUALITY, value=0.01)
_G = ConstraintSpec("g", upper=0.0)
_H = ConstraintSpec("h", lower=-0.1, upper=0.1)


def test_bencher_problem_returns_raw_constraints() -> None:
    """Constraints should be returned un-negated, alongside negated objectives."""
    problem = _bencher_problem((_G, _H), [_INEQ, _EQ])
    data = problem.evaluate_points(np.zeros((2, D)))
    np.testing.assert_array_equal(data.objectives, [[-2.0], [-2.0]])
    assert tuple(data.constraints) == ("g", "h")
    np.testing.assert_array_equal(data.constraints["g"], [-1.5, -1.5])
    np.testing.assert_array_equal(data.constraints["h"], [0.01, 0.01])


def test_bencher_problem_unconstrained() -> None:
    """With no constraints in the metadata or the result, the data is unconstrained."""
    problem = _bencher_problem((), [])
    data = problem.evaluate_points(np.zeros((2, D)))
    assert data.constraints == {}


def test_bencher_problem_constraints_without_specs_raises() -> None:
    """Constraints returned for a problem whose metadata defines none are an error."""
    problem = _bencher_problem((), [_INEQ])
    with pytest.raises(ValueError, match=r"unexpected \['g'\]"):
        problem.evaluate_points(np.zeros((1, D)))


def test_bencher_problem_constraint_count_mismatch_raises() -> None:
    """Bencher must return one constraint per spec in the metadata."""
    problem = _bencher_problem((_G, _H), [_INEQ])
    with pytest.raises(ValueError, match=r"missing \['h'\]"):
        problem.evaluate_points(np.zeros((1, D)))


def test_bencher_problem_constraint_name_mismatch_raises() -> None:
    """Constraints with names not matching the metadata are an error."""
    problem = _bencher_problem((ConstraintSpec("other", upper=0.0),), [_INEQ])
    with pytest.raises(ValueError, match=r"missing \['other'\], unexpected \['g'\]"):
        problem.evaluate_points(np.zeros((1, D)))


def test_bencher_problem_duplicate_constraint_names_raise() -> None:
    """Duplicate constraint names from Bencher are an error."""
    problem = _bencher_problem((_G, _H), [_INEQ, _INEQ])
    with pytest.raises(ValueError, match="duplicate constraint names"):
        problem.evaluate_points(np.zeros((1, D)))


def test_bencher_problem_equality_needs_two_sided_bounds() -> None:
    """An equality constraint without a tolerance on both sides is rejected."""
    problem = _bencher_problem((ConstraintSpec("h", upper=0.1),), [_EQ])
    with pytest.raises(ValueError, match="equality constraint"):
        problem.evaluate_points(np.zeros((1, D)))
