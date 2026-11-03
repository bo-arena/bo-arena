r"""Build CSV tables from raw ``*_ys.npy`` results for Harvard Dataverse.

Reads ``{problem}/{agent}/seed_{SEED}_ys.npy`` files (nested ``batch_{N}``
directories are ignored) and writes ``results.csv`` (one row per evaluation),
``problems.csv`` (one row per problem) into ``--out-dir``.

Usage:
    # Build the dataset, keeping only the problems and agents of interest:
    uv run python scripts/build_results_tables.py build \
        --results-root results \
        --out-dir dataverse \
        --problems "rover,mopta08,lasso-dna,svm,mujoco-ant,mujoco-humanoid" \
        --agents "random,turbo,saasbo,bounce,vanilla_bo,linear_bo,ibnn_depth_3"

    # Merge results into an existing table. Rows for any new problems are
    # added to the problems table so that it still covers every result.
    uv run python scripts/build_results_tables.py append \
        --existing dataverse/results.csv \
        --new new_results.csv \
        --out dataverse/results.csv \
        --problems-table dataverse/problems.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bo_arena.problems.metadata import PROBLEM_METADATA, SYNTHETIC_PROBLEM_METADATA

ALL_PROBLEM_METADATA = {**PROBLEM_METADATA, **SYNTHETIC_PROBLEM_METADATA}
SEED_FILE_RE = re.compile(r"^seed_(\d+)_ys\.npy$")

RUN_KEY = ["problem", "agent", "seed"]

# Stable column order + human-readable description for each output table.
RESULTS_COLUMNS: dict[str, str] = {
    "problem": "Problem name; joins to problems.csv.",
    "agent": "BO agent that produced the run",
    "seed": "Random seed identifying a single optimisation run.",
    "iteration": "0-based index of the function evaluation within the run.",
    "y": "Observed black-box value.",
}
PROBLEMS_COLUMNS: dict[str, str] = {
    "problem": "Problem name (primary key).",
    "input_dim": "Dimensionality of the problem input space.",
    "n_initial_points": "Initial design points evaluated before the BO loop.",
    "n_total_evaluations": "Total function evaluations in a run.",
    "stochastic": (
        "Whether the problem is stochastic, i.e. repeated evaluations at the "
        "same input can return different y values."
    ),
}


def _comma_separated_set(value: str) -> set[str]:
    """Parse a comma-separated CLI value into a set of non-empty tokens."""
    return {token.strip() for token in value.split(",") if token.strip()}


def results_tree_to_dataframe(
    results_root: Path,
    *,
    problems: set[str] | None = None,
    agents: set[str] | None = None,
) -> pd.DataFrame:
    """Walk a results tree and return the tidy long-format results table.

    Only files directly inside ``{problem}/{agent}/`` are read.

    Args:
        results_root: Directory containing
            ``{problem}/{agent}/seed_{SEED}_ys.npy`` files.
        problems: If given, keep only these problems.
        agents: If given, keep only these agents.

    Returns:
        A DataFrame with the columns of :data:`RESULTS_COLUMNS`.
    """
    frames: list[pd.DataFrame] = []
    for path in sorted(results_root.glob("*/*/seed_*_ys.npy")):
        match = SEED_FILE_RE.match(path.name)
        if match is None:
            continue
        problem, agent = path.relative_to(results_root).parts[:2]
        if problems is not None and problem not in problems:
            continue
        if agents is not None and agent not in agents:
            continue
        ys = np.load(path).astype(np.float64).ravel()
        frames.append(
            pd.DataFrame({
                "problem": problem,
                "agent": agent,
                "seed": np.int64(match.group(1)),
                "iteration": np.arange(ys.shape[0], dtype=np.int32),
                "y": ys,
            })
        )
    if not frames:
        msg = (
            f"No matching '{{problem}}/{{agent}}/seed_*_ys.npy' files found "
            f"under {results_root}"
        )
        raise FileNotFoundError(msg)
    return pd.concat(frames, ignore_index=True)[list(RESULTS_COLUMNS)]


def problem_metadata_to_dataframe(problems: list[str]) -> pd.DataFrame:
    """Build the per-problem metadata table for the given problems.

    Raises:
        ValueError: If any problem has no entry in the problem metadata.
    """
    unknown = sorted(set(problems) - ALL_PROBLEM_METADATA.keys())
    if unknown:
        msg = (
            f"No problem metadata registered for: {', '.join(unknown)}. Register "
            "them in bo_arena/problems/metadata.py."
        )
        raise ValueError(msg)
    rows: list[dict[str, Any]] = [
        {
            "problem": problem,
            "input_dim": int(ALL_PROBLEM_METADATA[problem].input_dim),
            "n_initial_points": int(
                ALL_PROBLEM_METADATA[problem].default_n_initial_points
            ),
            "n_total_evaluations": int(
                ALL_PROBLEM_METADATA[problem].default_n_total_evaluations
            ),
            "stochastic": bool(ALL_PROBLEM_METADATA[problem].stochastic),
        }
        for problem in problems
    ]
    return pd.DataFrame(rows, columns=list(PROBLEMS_COLUMNS))


def append_results(
    existing: Path, new: Path, out: Path, problems_table: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate a new results table onto an existing one, checking keys.

    The problems table at ``problems_table`` is updated in place with rows for
    any problems in the merged results that it does not already cover;
    existing rows are left untouched. Nothing is written if any check fails.

    Returns:
        The merged results table and the updated problems table.

    Raises:
        ValueError: If the schemas differ, the two tables share any run
            (a ``(problem, agent, seed)`` tuple) or a new problem has no
            registered metadata.
    """
    existing_df = pd.read_csv(existing)
    new_df = pd.read_csv(new)
    for label, frame in ((existing, existing_df), (new, new_df)):
        if list(frame.columns) != list(RESULTS_COLUMNS):
            msg = f"{label} does not match the expected results schema."
            raise ValueError(msg)
    overlap = (
        existing_df[RUN_KEY]
        .drop_duplicates()
        .merge(new_df[RUN_KEY].drop_duplicates(), on=RUN_KEY, how="inner")
    )
    if not overlap.empty:
        msg = (
            f"{len(overlap)} run(s) appear in both tables; refusing to create "
            f"duplicates. First few:\n{overlap.head().to_string(index=False)}"
        )
        raise ValueError(msg)
    merged = pd.concat([existing_df, new_df], ignore_index=True)

    problems_df = pd.read_csv(problems_table)
    if list(problems_df.columns) != list(PROBLEMS_COLUMNS):
        msg = f"{problems_table} does not match the expected problems schema."
        raise ValueError(msg)
    missing = sorted(set(merged["problem"]) - set(problems_df["problem"]))
    if missing:
        problems_df = pd.concat(
            [problems_df, problem_metadata_to_dataframe(missing)],
            ignore_index=True,
        ).sort_values("problem", ignore_index=True)

    merged.to_csv(out, index=False)
    problems_df.to_csv(problems_table, index=False)
    return merged, problems_df


def _cmd_build(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = results_tree_to_dataframe(
        args.results_root,
        problems=args.problems,
        agents=args.agents,
    )
    problems = sorted(results_df["problem"].unique())
    problems_df = problem_metadata_to_dataframe(problems)

    results_df.to_csv(out_dir / "results.csv", index=False)
    problems_df.to_csv(out_dir / "problems.csv", index=False)

    n_runs = results_df.groupby(RUN_KEY).ngroups
    print(f"Wrote {out_dir / 'results.csv'} ({len(results_df):,} rows, {n_runs} runs)")
    print(f"Wrote {out_dir / 'problems.csv'} ({len(problems_df)} problems)")


def _cmd_append(args: argparse.Namespace) -> None:
    merged, problems_df = append_results(
        args.existing, args.new, args.out, args.problems_table
    )
    print(f"Wrote {args.out} ({len(merged):,} rows)")
    print(f"Wrote {args.problems_table} ({len(problems_df)} problems)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build CSV tables from a results tree.")
    build.add_argument("--results-root", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)
    build.add_argument(
        "--problems",
        type=_comma_separated_set,
        default=None,
        help="Comma-separated problems to keep (default: all found in the tree).",
    )
    build.add_argument(
        "--agents",
        type=_comma_separated_set,
        default=None,
        help="Comma-separated agents to keep (default: all found in the tree).",
    )
    build.set_defaults(func=_cmd_build)

    append = sub.add_parser("append", help="Append a new results table onto another.")
    append.add_argument("--existing", type=Path, required=True)
    append.add_argument("--new", type=Path, required=True)
    append.add_argument("--out", type=Path, required=True)
    append.add_argument(
        "--problems-table",
        type=Path,
        required=True,
        help="problems.csv to update in place with rows for any new problems.",
    )
    append.set_defaults(func=_cmd_append)

    return parser


def main() -> None:
    """Entry point."""
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
