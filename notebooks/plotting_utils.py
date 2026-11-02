"""Utilities for plotting cumulative-max BO results in NeurIPS style."""

import functools
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import EngFormatter, MaxNLocator
from tueplots import figsizes, fonts, fontsizes
from tueplots.constants.color import palettes

from bo_arena.problems.metadata import PROBLEM_METADATA, SYNTHETIC_PROBLEM_METADATA
from bo_arena.experiment_seeds import EXPERIMENT_SEEDS
from bo_arena.utils.data_processing import get_cum_max_y_stats

PROBLEM_METADATA = {**PROBLEM_METADATA, **SYNTHETIC_PROBLEM_METADATA}

plt.rcParams.update({"figure.dpi": 200})

AGENT_NAME_MAPPING = {
    "bounce": "Bounce",
    "ibnn_depth_3": "IBNN",
    "linear_bo": "Linear BO",
    "random": "Random",
    "saasbo": "SAASBO",
    "turbo": "TuRBO",
    "vanilla_bo": "Vanilla BO",
}

_PAULTOL = palettes.paultol_bright
AGENT_COLOR_MAPPING = {
    "bounce": f"#{_PAULTOL[3]}",
    "ibnn_depth_3": f"#{_PAULTOL[4]}",
    "linear_bo": f"#{_PAULTOL[1]}",
    "random": "#555555",
    "saasbo": f"#{_PAULTOL[5]}",
    "turbo": f"#{_PAULTOL[2]}",
    "vanilla_bo": f"#{_PAULTOL[0]}",
}

AGENT_LINESTYLE_MAPPING = {
    "random": "dashed",
}

PROBLEM_NAME_MAPPING = {
    "ackley_dim_100": "Ackley (100D)",
    "michalewicz_dim_100": "Michalewicz (100D)",
    "styblinski_tang_dim_100": "Styblinski-Tang (100D)",
    "rover": "Rover (60D)",
    "mopta08": "MOPTA08 (124D)",
    "lasso-dna": "Lasso-DNA (180D)",
    "svm": "SVM (388D)",
    "mujoco-ant": "MuJoCo Ant (888D)",
    "mujoco-humanoid": "MuJoCo Humanoid (6392D)",
}

N_SEEDS = 20

_NOTEBOOKS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _NOTEBOOKS_DIR.parent / "results"
DATAVERSE_RESULTS_CSV = _NOTEBOOKS_DIR.parent / "dataverse" / "results.csv"
SAVEFIG_DIR = _NOTEBOOKS_DIR.parent / "figures"


@functools.cache
def _load_csv_runs(csv_path: Path) -> dict[tuple[str, str, int], np.ndarray]:
    """Index an exported results CSV by run.

    Args:
        csv_path: CSV with columns problem, algorithm, seed, iteration, y.

    Returns:
        A mapping from (problem, agent, seed) to that run's y values, ordered by
        iteration.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values(["problem", "algorithm", "seed", "iteration"])
    return {
        (problem, agent, int(seed)): group["y"].to_numpy(dtype=np.float64)
        for (problem, agent, seed), group in df.groupby(
            ["problem", "algorithm", "seed"], sort=False
        )
    }


def _load_run_ys(
    results_source: Path, problem: str, agent: str, seed: int
) -> np.ndarray | None:
    """Load the y values of a single run, or None if the run does not exist.

    Args:
        results_source: Either the root of a results directory tree laid out as
            `{results_source}/{problem}/{agent}/seed_{seed}_ys.npy`, or an exported
            results CSV.
        problem: Problem name.
        agent: Agent name.
        seed: Seed of the run.

    Returns:
        A 1D array of shape (n_evals,), or None if the run is not in the source.
    """
    if results_source.suffix == ".csv":
        return _load_csv_runs(results_source).get((problem, agent, seed))
    path = results_source / problem / agent / f"seed_{seed}_ys.npy"
    return np.load(path) if path.exists() else None


def _load_cum_max_stats(
    results_source: Path,
    problem: str,
    agent: str,
    seeds: list[int],
    allow_partial_seeds: bool,  # noqa: FBT001
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load runs and return (mean, sem) of the cumulative max.

    Emits a warning and returns None if the line should be skipped (either no
    runs found, or runs missing and partial seeds not allowed).

    Args:
        results_source: Results directory or exported results CSV
            (see `_load_run_ys`).
        problem: Problem name.
        agent: Agent name.
        seeds: Seeds to look for.
        allow_partial_seeds: If True, proceed with whatever seeds are available.
            If False, any missing seed causes the whole line to be skipped.

    Returns:
        A tuple of two 1D arrays of shape (n_evals,), or None if the line should
        be skipped:
            - The mean cumulative max y values across the available seeds.
            - The standard error of the mean of the cumulative max y values
              across the available seeds.
    """
    context = f"{agent} on {problem}"
    runs = [_load_run_ys(results_source, problem, agent, s) for s in seeds]
    available = [ys for ys in runs if ys is not None]
    missing = len(seeds) - len(available)

    if not available:
        warnings.warn(
            f"Skipping {context}: no runs found in {results_source}", stacklevel=3
        )
        return None
    if missing and not allow_partial_seeds:
        warnings.warn(
            f"Skipping {context}: {missing}/{len(seeds)} seeds missing",
            stacklevel=3,
        )
        return None
    if missing:
        warnings.warn(
            f"{context}: using {len(available)}/{len(seeds)} seeds ({missing} missing)",
            stacklevel=3,
        )

    return get_cum_max_y_stats(np.array(available))


def plot_cum_max_y(
    agents: list[str],
    problems: list[str],
    n_seeds: int = N_SEEDS,
    ncols: int | None = None,
    height_to_width_ratio: float = 0.95,
    agent_name_mapping: dict[str, str] | None = None,
    legend_ncols: int | None = None,
    filename: str | None = None,
    allow_partial_seeds: bool = True,
    overlay_last: list[str] | None = None,
    results_source: Path = RESULTS_DIR,
) -> None:
    """Plot cumulative maximum y values (mean +- 1SEM) for each agent on each problem.

    Results are read from `results_source`, which defaults to the
    `RESULTS_DIR/{problem}/{agent}/seed_{seed}_ys.npy` tree.

    Args:
        agents: Agent names (i.e. result subdirectory names) to plot.
        problems: Problems to plot, one subplot each.
        n_seeds: Number of seeds to consider.
        ncols: Number of columns in the subplot grid. Defaults to len(problems) (single row).
        height_to_width_ratio: Height-to-width ratio of each subplot.
        agent_name_mapping: Optional overrides for agent display names.
            Only needs to contain the names you want to change.
        legend_ncols: Number of columns in the legend. Defaults to one row (all entries).
        filename: Optional filename (without extension) to save the figure as PDF under SAVEFIG_DIR.
        allow_partial_seeds: If True, plot lines even when some runs are missing
            (uses only the available seeds). If False, skip any agent/problem
            combination missing seeds.
        overlay_last: Agent names that should be rendered on top of all other lines
            (via higher zorder). Legend order is unaffected.
        results_source: Where to read results from: either the root of a directory tree
            laid out as `{results_source}/{problem}/{agent}/seed_{seed}_ys.npy`, or an
            exported results CSV (columns problem, algorithm, seed, iteration, y).
    """
    # Merge user-provided name overrides on top of the defaults
    names = {**AGENT_NAME_MAPPING, **(agent_name_mapping or {})}
    seeds = EXPERIMENT_SEEDS[:n_seeds]
    # Sort problems by input dimensionality (ascending) so the grid reads left-to-right,
    # low-to-high regardless of the order callers pass them in
    problems = sorted(problems, key=lambda p: PROBLEM_METADATA[p].input_dim)

    # Compute grid shape (default: single row with one column per problem)
    n_problems = len(problems)
    ncols = ncols or n_problems
    nrows = math.ceil(n_problems / ncols)

    # Compose tueplots style: figure sizing + NeurIPS fonts and font sizes
    size = figsizes.neurips2024(
        ncols=ncols, nrows=nrows, height_to_width_ratio=height_to_width_ratio
    )
    style = {**size, **fonts.neurips2024_tex(), **fontsizes.neurips2024()}

    with plt.rc_context(style):
        fig, axes = plt.subplots(nrows, ncols, sharey=False, squeeze=False)
        axes_flat = axes.flatten()

        for i, (ax, problem) in enumerate(zip(axes_flat, problems)):
            n_init = PROBLEM_METADATA[problem].default_n_initial_points
            # Track post-init y-bounds across lines to auto-zoom the y-axis later
            post_init_mins = []
            post_init_maxs = []

            for agent_name in agents:
                stats = _load_cum_max_stats(
                    results_source, problem, agent_name, seeds, allow_partial_seeds
                )
                if stats is None:
                    continue
                cum_max_y_mean, cum_max_y_sem = stats
                iters = range(1, len(cum_max_y_mean) + 1)

                color = AGENT_COLOR_MAPPING[agent_name]
                ls = AGENT_LINESTYLE_MAPPING.get(agent_name, "solid")
                on_top = overlay_last and agent_name in overlay_last

                # Mean line + SEM band
                ax.plot(
                    iters, cum_max_y_mean, label=names[agent_name], color=color,
                    linestyle=ls, **({"zorder": 3} if on_top else {}),
                )
                ax.fill_between(
                    iters,
                    cum_max_y_mean - cum_max_y_sem,
                    cum_max_y_mean + cum_max_y_sem,
                    alpha=0.2,
                    color=color,
                    **({"zorder": 2.5} if on_top else {}),
                )

                # Record post-init bounds so the y-axis can ignore init-phase spikes
                post_init_data = slice(n_init, None)
                post_init_mins.append(
                    min((cum_max_y_mean - cum_max_y_sem)[post_init_data])
                )
                post_init_maxs.append(
                    max((cum_max_y_mean + cum_max_y_sem)[post_init_data])
                )

            # Dotted vertical line marking the end of the init phase
            ax.axvline(n_init + 0.5, color="gray", linestyle=(0, (1, 1)), alpha=0.5)

            # Zoom y-axis to the post-init range with a small margin
            if post_init_mins:
                y_min = min(post_init_mins)
                y_max = max(post_init_maxs)
                if y_max > y_min:
                    margin = 0.05 * (y_max - y_min)
                    ax.set_ylim(y_min - margin, y_max + margin)

            # Axis labels only on the outer edges of the grid
            ax.set_title(f"{PROBLEM_NAME_MAPPING[problem]}")
            row = i // ncols
            if row == nrows - 1:
                ax.set_xlabel("Iteration")
            if i % ncols == 0:
                ax.set_ylabel("Max. Observed Value")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            if ax.get_xlim()[1] >= 5000:
                ax.xaxis.set_major_formatter(EngFormatter(sep=""))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.grid(True, alpha=0.2)

        # Hide any leftover axes in the last (partially filled) row
        for j in range(n_problems, nrows * ncols):
            axes_flat[j].set_visible(False)

        fig.align_ylabels(axes[:, 0])

        # Single shared legend below the figure (dedupe across all axes so agents
        # absent from the first subplot still appear)
        seen = {}
        for ax in axes_flat[:n_problems]:
            for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
                if label not in seen:
                    seen[label] = handle
        fig.legend(
            list(seen.values()),
            list(seen.keys()),
            loc="outside lower center",
            ncols=legend_ncols or len(seen),
        )

        if filename is not None:
            SAVEFIG_DIR.mkdir(exist_ok=True)
            plt.savefig(SAVEFIG_DIR / f"{filename}.pdf")
        plt.show()
