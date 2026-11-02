r"""Build CSV tables from raw ``*_ys.npy`` results for Harvard Dataverse.

Reads ``{problem}/{algorithm}/seed_{SEED}_ys.npy`` files (nested ``batch_{N}``
directories are ignored) and writes ``results.csv`` (one row per evaluation),
``benchmarks.csv`` (one row per problem) into ``--out-dir``.

Usage:
    # Build the dataset, keeping only the problems and algorithms of interest:
    uv run python scripts/build_results_tables.py build \
        --results-root results \
        --out-dir dataverse \
        --problems "rover,mopta08,lasso-dna,svm,mujoco-ant,mujoco-humanoid" \
        --algorithms "random,turbo,saasbo,bounce,vanilla_bo,linear_bo,ibnn_depth_3"

    # Merge results into an existing table.
    uv run python scripts/build_results_tables.py append \
        --existing dataverse/results.csv \
        --new new_results.csv \
        --out dataverse/results.csv
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

RUN_KEY = ["problem", "algorithm", "seed"]

# Stable column order + human-readable description for each output table.
RESULTS_COLUMNS: dict[str, str] = {
    "problem": "Benchmark name; joins to benchmarks.csv.",
    "algorithm": "Optimisation algorithm that produced the run.",
    "seed": "Random seed identifying a single optimisation run.",
    "iteration": "0-based index of the function evaluation within the run.",
    "y": "Observed black-box value.",
}
BENCHMARKS_COLUMNS: dict[str, str] = {
    "problem": "Benchmark name (primary key).",
    "input_dim": "Dimensionality of the benchmark input space.",
    "n_initial_points": "Initial design points evaluated before the BO loop.",
    "n_total_evaluations": "Total function evaluations in a run.",
    "stochastic": (
        "Whether the benchmark is stochastic, i.e. repeated evaluations at the "
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
    algorithms: set[str] | None = None,
) -> pd.DataFrame:
    """Walk a results tree and return the tidy long-format results table.

    Only files directly inside ``{problem}/{algorithm}/`` are read.

    Args:
        results_root: Directory containing
            ``{problem}/{algorithm}/seed_{SEED}_ys.npy`` files.
        problems: If given, keep only these problems.
        algorithms: If given, keep only these algorithms.

    Returns:
        A DataFrame with the columns of :data:`RESULTS_COLUMNS`.
    """
    frames: list[pd.DataFrame] = []
    for path in sorted(results_root.glob("*/*/seed_*_ys.npy")):
        match = SEED_FILE_RE.match(path.name)
        if match is None:
            continue
        problem, algorithm = path.relative_to(results_root).parts[:2]
        if problems is not None and problem not in problems:
            continue
        if algorithms is not None and algorithm not in algorithms:
            continue
        ys = np.load(path).astype(np.float64).ravel()
        frames.append(
            pd.DataFrame({
                "problem": problem,
                "algorithm": algorithm,
                "seed": np.int64(match.group(1)),
                "iteration": np.arange(ys.shape[0], dtype=np.int32),
                "y": ys,
            })
        )
    if not frames:
        msg = (
            f"No matching '{{problem}}/{{algorithm}}/seed_*_ys.npy' files found "
            f"under {results_root}"
        )
        raise FileNotFoundError(msg)
    return pd.concat(frames, ignore_index=True)[list(RESULTS_COLUMNS)]


def benchmark_metadata_to_dataframe(problems: list[str]) -> pd.DataFrame:
    """Build the per-benchmark metadata table for the given problems."""
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
    return pd.DataFrame(rows, columns=list(BENCHMARKS_COLUMNS))


def append_results(existing: Path, new: Path, out: Path) -> pd.DataFrame:
    """Concatenate a new results table onto an existing one, checking keys.

    Raises:
        ValueError: If the schemas differ or the two tables share any run
            (a ``(problem, algorithm, seed)`` tuple).
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
    merged.to_csv(out, index=False)
    return merged


def _cmd_build(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = results_tree_to_dataframe(
        args.results_root,
        problems=args.problems,
        algorithms=args.algorithms,
    )
    problems = sorted(results_df["problem"].unique())
    benchmarks_df = benchmark_metadata_to_dataframe(problems)

    results_df.to_csv(out_dir / "results.csv", index=False)
    benchmarks_df.to_csv(out_dir / "benchmarks.csv", index=False)

    n_runs = results_df.groupby(RUN_KEY).ngroups
    print(f"Wrote {out_dir / 'results.csv'} ({len(results_df):,} rows, {n_runs} runs)")
    print(f"Wrote {out_dir / 'benchmarks.csv'} ({len(benchmarks_df)} benchmarks)")
    print(f"Wrote {out_dir / 'README.md'}")


def _cmd_append(args: argparse.Namespace) -> None:
    merged = append_results(args.existing, args.new, args.out)
    print(f"Wrote {args.out} ({len(merged):,} rows)")


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
        "--algorithms",
        type=_comma_separated_set,
        default=None,
        help="Comma-separated algorithms to keep (default: all found in the tree).",
    )
    build.set_defaults(func=_cmd_build)

    append = sub.add_parser("append", help="Append a new results table onto another.")
    append.add_argument("--existing", type=Path, required=True)
    append.add_argument("--new", type=Path, required=True)
    append.add_argument("--out", type=Path, required=True)
    append.set_defaults(func=_cmd_append)

    return parser


def main() -> None:
    """Entry point."""
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
