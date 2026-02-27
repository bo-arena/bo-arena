"""Tests for randomness."""

import numpy as np
import pytest

from bo_arena.agents.registry import AGENTS
from bo_arena.arena import Arena
from bo_arena.benchmarks.synthetic_benchmarks import Hartmann6D
from bo_arena.utils.dataset import Dataset
from bo_arena.utils.miscellaneous import generate_seed, split_np


@pytest.mark.parametrize("agent_name", AGENTS.keys())
def test_seed_determinism(agent_name: str) -> None:
    """Running the Arena twice with the same seed should give identical results."""
    benchmark = Hartmann6D(embed_dim=100)
    agent_cls = AGENTS[agent_name]

    def run_arena(seed: int) -> Dataset:
        agent = agent_cls(input_dim=benchmark.metadata.input_dim)
        arena = Arena(
            benchmark=benchmark,
            agent=agent,
            seed=seed,
            batch_size=1,
            n_initial_points=10,
            n_total_evaluations=12,
        )
        return arena.run()

    result_1 = run_arena(seed=42)
    result_2 = run_arena(seed=42)

    np.testing.assert_array_equal(result_1.X, result_2.X)
    np.testing.assert_array_equal(result_1.y, result_2.y)


@pytest.mark.parametrize("agent_name", AGENTS.keys())
def test_different_seeds_give_different_results(agent_name: str) -> None:
    """Running the Arena with different seeds should give different results."""
    benchmark = Hartmann6D(embed_dim=100)
    agent_cls = AGENTS[agent_name]

    def run_arena(seed: int) -> Dataset:
        agent = agent_cls(input_dim=benchmark.metadata.input_dim)
        arena = Arena(
            benchmark=benchmark,
            agent=agent,
            seed=seed,
            batch_size=1,
            n_initial_points=10,
            n_total_evaluations=12,
        )
        return arena.run()

    result_1 = run_arena(seed=42)
    result_2 = run_arena(seed=123)

    assert not np.array_equal(result_1.X, result_2.X)


def test_seed_splitting_is_deterministic() -> None:
    """Splitting a SeedSequence must return the same values every time."""
    for _ in range(5):
        key = np.random.SeedSequence(42)
        key, subkey = split_np(key, n=2)
        seed_a = generate_seed(subkey)

        key = np.random.SeedSequence(42)
        key, subkey = split_np(key, n=2)
        seed_b = generate_seed(subkey)

        assert seed_a == seed_b


@pytest.mark.parametrize(
    ("master_seed", "expected_seeds"),
    [
        (0, [673228719, 3581274545, 1745761672]),
        (42, [4091952314, 3908812709, 2424979062]),
        (123456, [2431930328, 3878087563, 2539469541]),
    ],
)
def test_seed_splitting_hard_coded_values(
    master_seed: int, expected_seeds: list[int]
) -> None:
    """Guard against SeedSequence splitting changing across NumPy versions."""
    key = np.random.SeedSequence(master_seed)
    for expected in expected_seeds:
        key, subkey = split_np(key, n=2)
        assert generate_seed(subkey) == expected
