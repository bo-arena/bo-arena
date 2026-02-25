"""Tests that agents produce deterministic results when given the same seed."""

import numpy as np
import pytest

from bo_arena.agents.registry import AGENTS
from bo_arena.arena import Arena
from bo_arena.benchmarks.synthetic_benchmarks import Hartmann6D
from bo_arena.utils.dataset import Dataset


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
        )
        return arena.run()

    result_1 = run_arena(seed=42)
    result_2 = run_arena(seed=123)

    assert not np.array_equal(result_1.X, result_2.X)
