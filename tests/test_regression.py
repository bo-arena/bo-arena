"""Regression tests for BO agent performance.

Compares all registered agents against checked-in performance baselines to detect
unintended changes caused by dependency bumps or code modifications.

Run with:
    uv run pytest -m regression -s
"""

import datetime
import operator
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float

from bo_arena.agents.registry import AGENTS
from bo_arena.arena import Arena
from bo_arena.benchmarks.metadata import SYNTHETIC_BENCHMARK_METADATA
from bo_arena.benchmarks.synthetic_benchmarks import Hartmann6D
from bo_arena.experiment_seeds import EXPERIMENT_SEEDS

# ── Regression configuration ──────────────────────────────────────────────────
BUDGET_PERCENTAGES: list[int] = [10, 20, 50, 100]
BENCHMARK_NAME = "hartmann6d_embed_dim_300"
BENCHMARK_META = SYNTHETIC_BENCHMARK_METADATA[BENCHMARK_NAME]
N_TOTAL_EVALUATIONS: int = BENCHMARK_META.default_n_total_evaluations
N_INITIAL_POINTS: int = BENCHMARK_META.default_n_initial_points

BASELINES_DIR = Path(__file__).parent / "regression_baselines"
REPORTS_DIR = Path(__file__).parent / "regression_reports"


def run_agent(agent_name: str, seed: int) -> np.ndarray:
    """Run a single Arena trial and return the y-values.

    Args:
        agent_name: Name of the agent to run.
        seed: Random seed for this trial.

    Returns:
        Observed y-values.
    """
    benchmark = Hartmann6D(embed_dim=300)
    agent = AGENTS[agent_name](
        input_dim=benchmark.metadata.input_dim,
        n_total_evaluations=N_TOTAL_EVALUATIONS,
    )
    arena = Arena(
        benchmark=benchmark,
        agent=agent,
        seed=seed,
        batch_size=1,
    )
    return arena.run().observed_data.y


def best_at_pct(
    y_matrix: Float[np.ndarray, "n_seeds n_evals"], pct: int
) -> Float[np.ndarray, " n_seeds"]:
    """Return the best-so-far value at pct% of budget for each seed row.

    Args:
        y_matrix: y-values observed for each seeded run.
        pct: Budget percentage in [1, 100].

    Returns:
        Array of best values seen up to pct% of the budget.
    """
    n_evals = y_matrix.shape[1]
    cutoff = max(1, int(np.ceil(n_evals * pct / 100)))
    return np.max(y_matrix[:, :cutoff], axis=1)


def cell(mean: float, std: float) -> str:
    """Format a mean \u00b1 std pair as a fixed-width report column.

    Args:
        mean: Mean value.
        std: Standard deviation.

    Returns:
        Fixed-width string of the form "{mean} \u00b1 {std}" (20 characters).
    """
    return f"{mean:9.4f} \u00b1 {std:8.4f}"


def build_report(
    all_current: dict[str, Float[np.ndarray, "n_seeds n_evals"]],
    all_baselines: dict[str, Float[np.ndarray, "n_seeds n_evals"] | None],
) -> str:
    """Build the full regression report string.

    Args:
        all_current: Maps agent name to y-value matrix containing y-values observed for
            each seeded run.
        all_baselines: Maps agent name to baseline y-value matrix or None if
            unavailable.

    Returns:
        Formatted regression report.
    """
    sep = "=" * 80
    n_bo = N_TOTAL_EVALUATIONS - N_INITIAL_POINTS
    lines: list[str] = [
        sep,
        f"REGRESSION REPORT: {BENCHMARK_NAME}",
        (
            f"{len(EXPERIMENT_SEEDS)} seeds"
            f" | Budget: {N_TOTAL_EVALUATIONS} total evaluations"
            f" ({N_INITIAL_POINTS} initial + {n_bo} BO)"
        ),
        sep,
        "",
    ]

    best_at_100: dict[str, float] = {}

    for agent_name in sorted(all_current):
        current = all_current[agent_name]
        baseline = all_baselines.get(agent_name)

        lines.append(f"AGENT: {agent_name}")

        if baseline is None:
            lines.extend([
                (
                    "  Baseline: not available"
                    " \u2014 run scripts/generate_regression_baselines.py to create it."
                ),
                "",
                f"  {'Budget %':>9}  {'Current mean \u00b1 std'}",
                "  " + "-" * 36,
            ])
            for pct in BUDGET_PERCENTAGES:
                curr_vals = best_at_pct(current, pct)
                lines.append(
                    f"  {pct:>8d}%"
                    f"  {cell(float(curr_vals.mean()), float(curr_vals.std()))}"
                )
        else:
            identical = bool(np.array_equal(current, baseline))
            lines.extend([
                (
                    f"  Baseline: available"
                    f" | All runs identical: {'YES' if identical else 'NO'}"
                ),
                "",
                (
                    f"  {'Budget %':>9}"
                    f"  {'Baseline mean \u00b1 std':<20}"
                    f"  {'Current mean \u00b1 std':<20}"
                    f"  {'delta mean':>10}"
                ),
                "  " + "-" * 67,
            ])
            for pct in BUDGET_PERCENTAGES:
                base_vals = best_at_pct(baseline, pct)
                curr_vals = best_at_pct(current, pct)
                delta = float(curr_vals.mean()) - float(base_vals.mean())
                lines.append(
                    f"  {pct:>8d}%"
                    f"  {cell(float(base_vals.mean()), float(base_vals.std()))}"
                    f"  {cell(float(curr_vals.mean()), float(curr_vals.std()))}"
                    f"  {delta:>+10.4f}"
                )

            base_100 = best_at_pct(baseline, 100)
            curr_100 = best_at_pct(current, 100)
            n_better = int(np.sum(curr_100 > base_100))
            n_same = int(np.sum(curr_100 == base_100))
            n_worse = int(np.sum(curr_100 < base_100))
            lines.extend([
                "",
                (
                    f"  Runs at 100% budget \u2014"
                    f" better: {n_better} | same: {n_same} | worse: {n_worse}"
                ),
            ])

        best_at_100[agent_name] = float(best_at_pct(current, 100).mean())
        lines.append("")

    ranked = sorted(best_at_100.items(), key=operator.itemgetter(1), reverse=True)
    lines.extend([
        sep,
        "RANKING at 100% budget (mean best observed value, current run)",
        sep,
    ])
    for rank, (name, val) in enumerate(ranked, start=1):
        lines.append(f"  {rank}. {name}: {val:.4f}")
    lines.extend(["", sep])

    return "\n".join(lines)


@pytest.mark.regression
def test_regression(request: pytest.FixtureRequest) -> None:
    """Compare registered agents against checked-in performance baselines.

    This test always passes — its purpose is to produce a report showing whether agent
    performance has changed relative to the baselines stored in
    tests/regression_baselines/.

    Run with:
        uv run pytest -m regression -s                         # all agents
        uv run pytest -m regression -s --agents random         # one agent
        uv run pytest -m regression -s --agents random gp_ucb  # subset

    The report is also written to tests/regression_reports/.
    """
    agent_filter: str | None = request.config.getoption("--agents")
    agents_to_run: list[str] = (
        [a.strip() for a in agent_filter.split(",")]
        if agent_filter is not None
        else list(AGENTS.keys())
    )

    unknown = [a for a in agents_to_run if a not in AGENTS]
    if unknown:
        pytest.fail(f"Unknown agent(s): {unknown}. Available: {list(AGENTS.keys())}")

    all_current: dict[str, np.ndarray] = {}
    all_baselines: dict[str, np.ndarray | None] = {}

    for agent_name in agents_to_run:
        all_current[agent_name] = np.array([
            run_agent(agent_name, seed) for seed in EXPERIMENT_SEEDS
        ])
        baseline_path = BASELINES_DIR / BENCHMARK_NAME / f"{agent_name}.npz"
        if baseline_path.exists():
            data = np.load(baseline_path)
            assert np.array_equal(data["seeds"], np.array(EXPERIMENT_SEEDS)), (
                f"Seeds in baseline for '{agent_name}' do not match EXPERIMENT_SEEDS."
                " Regenerate baselines with scripts/generate_regression_baselines.py."
            )
            expected_shape = (len(EXPERIMENT_SEEDS), N_TOTAL_EVALUATIONS)
            assert data["y_values"].shape == expected_shape, (
                f"Baseline shape for '{agent_name}' is {data['y_values'].shape},"
                f" expected {expected_shape}. The benchmark metadata may have changed."
                " Regenerate baselines with scripts/generate_regression_baselines.py."
            )
            all_baselines[agent_name] = data["y_values"]
        else:
            all_baselines[agent_name] = None

    report = build_report(all_current, all_baselines)

    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"report_{timestamp}.txt"
    report_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nReport also written to: {report_path}")
