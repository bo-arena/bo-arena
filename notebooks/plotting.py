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
from plotting_utils import plot_cum_max_y

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
    agents=["random", "turbo", "bounce", "vanilla_bo", "linear_bo", "ibnn_depth_3"],
    problems=["ackley_dim_100", "michalewicz_dim_100", "styblinski_tang_dim_100"],
    legend_ncols=3,
    filename="synthetic_problems",
)
# %%
