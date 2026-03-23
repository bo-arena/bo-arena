"""Script to generate or update regression test baselines.

Baselines are stored as .npz files under tests/regression_baselines/ and should be
committed to the repository. Re-run this script whenever an intentional change to an
agent's behaviour warrants updating the reference results.

Usage:
    # Generate baselines for all registered agents (skip if already exist):
    uv run python scripts/generate_regression_baselines.py

    # Regenerate only the random agent's baseline:
    uv run python scripts/generate_regression_baselines.py --agents random

    # Regenerate baselines for multiple agents:
    uv run python scripts/generate_regression_baselines.py --agents random,gp_ucb

    # Overwrite any existing baselines:
    uv run python scripts/generate_regression_baselines.py --force
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from bo_arena.agents.registry import AGENTS
from bo_arena.arena import Arena
from bo_arena.benchmarks.metadata import SYNTHETIC_BENCHMARK_METADATA
from bo_arena.benchmarks.synthetic_benchmarks import Hartmann6D
from bo_arena.experiment_seeds import EXPERIMENT_SEEDS

# ── Regression configuration ──────────────────────────────────────────────────
BENCHMARK_NAME = "hartmann6d_embed_dim_300"
BASELINES_DIR = Path(__file__).parent.parent / "tests" / "regression_baselines"


def run_agent(agent_name: str, seed: int) -> np.ndarray:
    """Run a single Arena trial and return the y-values in evaluation order.

    Args:
        agent_name: Name of the agent to run.
        seed: Random seed for this trial.

    Returns:
        Observed y-values.
    """
    benchmark = Hartmann6D(embed_dim=300)
    agent = AGENTS[agent_name](input_dim=benchmark.metadata.input_dim)
    arena = Arena(
        benchmark=benchmark,
        agent=agent,
        seed=seed,
        batch_size=1,
    )
    return arena.run().observed_data.y


def generate_baselines(agents: list[str], *, force: bool) -> None:
    """Generate and save regression baselines for the specified agents.

    Args:
        agents: List of agent names to generate baselines for.
        force: If True, overwrite existing baseline files.
    """
    output_dir = BASELINES_DIR / BENCHMARK_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    for agent_name in agents:
        if agent_name not in AGENTS:
            print(
                f"Unknown agent '{agent_name}'. Available: {list(AGENTS.keys())}",
                file=sys.stderr,
            )
            continue

        output_path = output_dir / f"{agent_name}.npz"
        if output_path.exists() and not force:
            print(
                f"Baseline for '{agent_name}' already exists at {output_path}."
                " Use --force to overwrite."
            )
            continue

        n_evals = SYNTHETIC_BENCHMARK_METADATA[
            BENCHMARK_NAME
        ].default_n_total_evaluations
        n_seeds = len(EXPERIMENT_SEEDS)
        print(
            f"Generating baselines for '{agent_name}'"
            f" ({n_seeds} seeds, {n_evals} evaluations each)..."
        )
        y_values = np.array([run_agent(agent_name, seed) for seed in EXPERIMENT_SEEDS])
        np.savez(output_path, y_values=y_values, seeds=np.array(EXPERIMENT_SEEDS))
        print(f"  Saved to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate regression test baselines for BO agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--agents",
        type=lambda s: [a.strip() for a in s.split(",")],
        default=list(AGENTS.keys()),
        metavar="AGENT[,AGENT,...]",
        help=(
            "Comma-separated agents to generate baselines for (default: all)."
            f" Available: {list(AGENTS.keys())}"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing baseline files.",
    )
    return parser.parse_args()


def main() -> None:
    """Parse arguments and generate regression baselines."""
    args = parse_args()
    generate_baselines(agents=args.agents, force=args.force)


if __name__ == "__main__":
    main()
