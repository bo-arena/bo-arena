# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: bo-arena (3.14.8)
#     language: python
#     name: python3
# ---

# %%
from plotting_utils import DATAVERSE_RESULTS_CSV, plot_cum_max_y

# %% [markdown]
# ### Plot Results from CSV file
#
# Produce the plot from results stored as a CSV file (as in the Dataverse).

# %%
plot_cum_max_y(
    agents=["random", "turbo", "saasbo", "bounce", "vanilla_bo", "linear_bo", "ibnn_depth_3"],
    problems=["rover", "mopta08", "lasso-dna", "svm", "mujoco-ant", "mujoco-humanoid"],
    ncols=3,
    height_to_width_ratio=0.83,
    legend_ncols=4,
    filename=None,
    overlay_last=["saasbo"],
    results_source=DATAVERSE_RESULTS_CSV,
)

# %% [markdown]
# ### Plot Results from NumPy Arrays
#
# Produce the plot from results stored as NumPy arrays in the directory tree following the format `{RESULTS_DIR}/{problem}/{agent}/seed_{seed}_ys.npy`.

# %%
plot_cum_max_y(
    agents=["random", "turbo", "saasbo", "bounce", "vanilla_bo", "linear_bo", "ibnn_depth_3"],
    problems=["rover", "mopta08", "lasso-dna", "svm", "mujoco-ant", "mujoco-humanoid"],
    ncols=3,
    height_to_width_ratio=0.83,
    legend_ncols=4,
    filename="arena_leaderboard",
    overlay_last=["saasbo"],
)

# %%
plot_cum_max_y(
    agents=["random", "turbo", "bounce", "vanilla_bo", "linear_bo"],
    problems=["ackley_dim_100", "michalewicz_dim_100", "styblinski_tang_dim_100"],
    filename="synthetic_problems",
)

# %%
