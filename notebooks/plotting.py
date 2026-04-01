# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
from tueplots import bundles
from tueplots import cycler
from tueplots.constants.color import palettes
from bo_arena.utils.data_processing import get_cum_max_y_stats, get_all_seed_y_vals
from bo_arena.experiment_seeds import EXPERIMENT_SEEDS
from pathlib import Path
from bo_arena.benchmarks.metadata import BENCHMARK_METADATA

plt.rcParams.update({"figure.dpi": 200})

AGENT_NAME_MAPPING = {
    "baxus": "BAxUS",
    "linear_bo": "Linear BO",
    "random": "Random",
    "saasbo": "SAASBO",
    "turbo": "TuRBO",
    "vanilla_bo": "Vanilla BO",
}

PROBLEM_NAME_MAPPING = {
    "lasso-dna": "Lasso-DNA",
    "mopta08": "MOPTA08",
    "mujoco-ant": "MuJoCo Ant",
    "mujoco-humanoid": "MuJoCo Humanoid",
    "rover": "Rover",
    "svm": "SVM",
}

# AGENTS = ["baxus", "linear_bo", "random", "saasbo", "turbo", "vanilla_bo" ]
# PROBLEMS = ["lasso-dna", "mopta08", "mujoco-ant", "mujoco-humanoid", "rover", "svm"]
AGENTS = ["baxus", "linear_bo", "random", "turbo", "vanilla_bo" ]
PROBLEMS = ["lasso-dna", "mopta08", "svm"]
N_SEEDS = 10


# %%
def plot_cum_max_y(agents: list, problem: str, n_seeds: int, strategy: str):
    """Plots cumulative maximum y values (mean +- 1STD).

    Args:
        agents: List of agent names.
        problem: Problem name.
        n_seeds: Number of seeds to consider.
        strategy: Optimization strategy ("default", "raasp" or "no_raasp").
    """
    if strategy not in ["default", "raasp", "no_raasp"]:
        raise ValueError("Invalid strategy. Must be one of 'default', 'raasp', or 'no_raasp'.")

    n_init = BENCHMARK_METADATA[problem].default_n_initial_points
    seeds = EXPERIMENT_SEEDS[:n_seeds]

    with plt.rc_context({**bundles.neurips2024(), **cycler.cycler(color=palettes.tue_ai)}):
        for agent in agents:
            res_path = Path(f"../results/{problem}/{agent}/acqf_opt_{strategy}/")
            all_seed_ys = get_all_seed_y_vals(res_path, seeds)
            cum_max_y_mean, cum_max_y_std = get_cum_max_y_stats(all_seed_ys)
            iters = range(1, len(cum_max_y_mean) + 1)
            plt.plot(iters, cum_max_y_mean, label=AGENT_NAME_MAPPING[agent])
            plt.fill_between(iters, cum_max_y_mean - cum_max_y_std, cum_max_y_mean + cum_max_y_std, alpha=0.2)


        plt.axvline(n_init+0.5, color="gray", linestyle="--", alpha=0.5)
        plt.title(f"{PROBLEM_NAME_MAPPING[problem]}")
        plt.xlabel("Iteration")
        plt.ylabel("Max. Observed Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

# %% [markdown]
# ### Algorithms run with paper defaults

# %%
plot_cum_max_y(AGENTS, "svm", n_seeds=N_SEEDS, strategy="default")


# %% [markdown]
# ### RAASP vs No RAASP

# %%
def plot_cum_max_y_raasp_vs_no_raasp(agent: str, problem: str, n_seeds: int):
    """Plots cumulative maximum y values (mean +- 1STD) for RAASP vs No RAASP.

    Args:
        agent: Agent name.
        problem: Problem name.
        n_seeds: Number of seeds to consider.
    """
    n_init = BENCHMARK_METADATA[problem].default_n_initial_points
    dim = BENCHMARK_METADATA[problem].input_dim
    seeds = EXPERIMENT_SEEDS[:n_seeds]

    with plt.rc_context({**bundles.neurips2024(), **cycler.cycler(color=palettes.tue_ai)}):
        if agent == "turbo" or agent == "baxus":
            # For our experiments, default and RAASP strategies are the same for TuRBO
            # and BAxUS (since all problems are > 50D), so we just ran experiments for
            # the default strategy.
            if dim < 50:
                raise ValueError("Warning! Default strategy isn't exactly the same as RAASP for BAxUS and TuRBO on problems with dimensionality < 50, and the experimental setup wasn't written to run both strategies for these agents, due to no problems satisfying this condition.")
            raasp_res_path = Path(f"../results/{problem}/{agent}/acqf_opt_default/")
        else:
            raasp_res_path = Path(f"../results/{problem}/{agent}/acqf_opt_raasp/")
        no_raasp_res_path = Path(f"../results/{problem}/{agent}/acqf_opt_no_raasp/")
        raasp_ys = get_all_seed_y_vals(raasp_res_path, seeds)
        no_raasp_ys = get_all_seed_y_vals(no_raasp_res_path, seeds)
        raasp_cum_max_y_mean, raasp_cum_max_y_std = get_cum_max_y_stats(raasp_ys)
        no_raasp_cum_max_y_mean, no_raasp_cum_max_y_std = get_cum_max_y_stats(no_raasp_ys)
        iters = range(1, len(raasp_cum_max_y_mean) + 1)

        plt.plot(iters, raasp_cum_max_y_mean, label=f"{AGENT_NAME_MAPPING[agent]} (RAASP)")
        plt.plot(iters, no_raasp_cum_max_y_mean, label=f"{AGENT_NAME_MAPPING[agent]} (No RAASP)")
        plt.fill_between(iters, raasp_cum_max_y_mean - raasp_cum_max_y_std, raasp_cum_max_y_mean + raasp_cum_max_y_std, alpha=0.2)
        plt.fill_between(iters, no_raasp_cum_max_y_mean - no_raasp_cum_max_y_std, no_raasp_cum_max_y_mean + no_raasp_cum_max_y_std, alpha=0.2)

        plt.axvline(n_init + 0.5, color="gray", linestyle="--", alpha=0.5)
        plt.title(f"{PROBLEM_NAME_MAPPING[problem]}")
        plt.xlabel("Iteration")
        plt.ylabel("Max. Observed Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()


# %%
plot_cum_max_y_raasp_vs_no_raasp("turbo", "svm", N_SEEDS)
