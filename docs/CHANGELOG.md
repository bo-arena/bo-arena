# Changelog

## v1.0.0

Initial release of codebase to accompany the paper.

For running the experiments, we used the following Docker images available on Docker Hub:
- `bo-arena`: `docker://tchristie/bo-arena@sha256:e1eb20f52187709dd3f325e103cf48805fd48b18449366c91d112bdf02a403a1`
- `bencher`: `docker://tchristie/bencher@sha256:a7ca83f6d0b3810ed8dc03bb210aa35c46e65d5d19e4371aa7dbf038be3b6467`

These were used to create Singularity `.sif` files using the the `pull_bencher_singularity.sbatch` and `pull_bo_arena_singularity.sbatch` scripts. The exact `.sif` files used for running the experiments can be found [here](https://drive.google.com/drive/folders/1HCIabVFaGDb_4sDIJ_mUK9PBOEQKXhTC?usp=sharing).

### Experiment Commands

Below are the commands used to run the experiments for the leaderboard:

```
python slurm/submit.py \
--problem rover,mopta08,lasso-dna,svm,mujoco-ant,mujoco-humanoid \
--agent vanilla_bo,linear_bo,random,turbo,ibnn \
--seed-range 0-19
```

The bounce results are obtained via the following command (as the number of initial points used in the paper is 5, due to a low-dimensional subspace being used at the start):

```
python slurm/submit.py \
--problem rover,mopta08,lasso-dna,svm,mujoco-ant,mujoco-humanoid \
--agent bounce \
--n-init 5 \
--seed-range 0-19
```

The SAASBO results are obtained via the following command, as we limited the total number of evaluations to 500 due to the expense of running the algorithm (and didn't run on mujoco-humanoid as it was too expensive to run with SAASBO):

```
python slurm/submit.py \
--problem rover,mopta08,lasso-dna,svm,mujoco-ant \
--agent saasbo \
--n-evals 500 \
--seed-range 0-19
```

Code used for plotting the leaderboard figure can be found in `notebooks/plotting.py`.
