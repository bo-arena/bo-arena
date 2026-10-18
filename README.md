# bo-arena
Arena for simple, reliable, and efficient benchmarking of Bayesian optimisation algorithms. This repository contains implementations of state-of-the-art high-dimensional Bayesian optimisation algorithms, and code for running these algorithms on a variety of problems, primarily provided through [bencher](https://github.com/lpapenme/bencher). The code in this repository was also used to generate the leaderboard displayed on our website, and instructions for downloading the data in order to recreate the figures can be found there. Our accompanying paper, `BO-Arena: An Evolving Benchmark for High-Dimensional Bayesian Optimisation`, published in the Evaluations & Datasets track at NeurIPS 2026 (citation below), contains more details concerning the motivation for this repository, and additional ablations. The repository used for running the ablations in the paper can be found [here](https://github.com/bo-arena/bo-arena-neurips2026).

### Contributing

Please refer to the [contributing guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

### Quickstart

In order to run a quick experiment with `bo-arena` follow these steps:

1. Install dependencies using `uv` (see the "Install Dependencies" section of the [contributing guidelines](CONTRIBUTING.md) for details).
2. Start up the `bencher` server. You will need to first pull the docker image with:

    ```
    docker pull --platform linux/amd64 tchristie/bencher@sha256:a7ca83f6d0b3810ed8dc03bb210aa35c46e65d5d19e4371aa7dbf038be3b6467
    ```

    Then, you can start the server with:

    ```
    docker run --platform linux/amd64 -p 50051:50051 --restart always -d tchristie/bencher@sha256:a7ca83f6d0b3810ed8dc03bb210aa35c46e65d5d19e4371aa7dbf038be3b6467
    ```
    Note that this image was built from the `bencher` repository at commit `ba9ada07a7f3eaa5efd691200ed4866ff4ca6e1b`. If you want to create your own image, please refer to the instructions in [the bencher README](https://github.com/lpapenme/bencher).
3. Run the `quickstart` notebook in the `notebooks/` directory, which demonstrates how to run a simple experiment with `bo-arena` using the `bencher` server you just started. For instructions for converting between `.py` and `.ipynb` formats, see the "Notebooks" section of the [contributing guidelines](CONTRIBUTING.md).

### Running Experiments with bencher on a SLURM Cluster
The `slurm` directory contains SLURM batch scripts for building Singularity containers and running `bencher` and `bo-arena` experiments on a SLURM cluster. These scripts may require modifications to paths, resource specifications, and other parameters to fit your specific cluster setup.

Each experiment job is self-contained: it starts a `bencher` server as a sidecar process, runs the experiment, then shuts the server down on exit. This means no shared `bencher` server is needed, jobs scale independently, and there is no coordination between jobs. Each job gives its sidecar a private set of Unix domain sockets under `$TMPDIR/bencher-$SLURM_JOB_ID`, so jobs co-scheduled on the same node cannot reach each other's server. There are two containers involved per job:

1. **`bencher` container**: Runs the `bencher` server on the job's own sockets. Started as a background sidecar at the beginning of each experiment job and stopped automatically when the job exits.
2. **`bo-arena` experiment container**: Runs the `bo-arena` experiment, connecting to the `bencher` server over that job's router socket.

We use Unix sockets rather than fixed TCP ports to prevent different SLURM jobs from using each other's sidecars. TCP ports live in the node's network namespace, so jobs on the same node bind to the same TCP ports (50051, 50053-50060). In contrast, sockets are addressed by filesystem path, so putting the job id in the path means different jobs can't share each other's sidecar, avoiding this problem.

> **⚠️ Warning**
>
> Important - The run_experiment.sbatch script creates socket files for communicating with bencher under the directory $TMPDIR/bencher-$SLURM_JOB_ID (falling back to /tmp/bencher-$SLURM_JOB_ID if $TMPDIR is not set). The script generally removes this directory (and the created sockets) once it has finished running, but some cases (e.g. SIGKILL) it will not. Ideally $TMPDIR should be a job-specific temporary directory, so it will get deleted once the job finishes running in any case (on many clusters this is the case). If not, be careful, as $TMPDIR could get cluttered with directories containing socket files.

In order to run experiments, follow these steps:
1. Copy `slurm/config.sh.example` to `slurm/config.sh` and fill in your
cluster-specific paths, partition names and WandB credentials (if running with WandB,
as is the default behaviour). This file is gitignored.
2. Obtain the Singularity images, which are expected to be in an `images/` directory.
    - **`bencher`**: pull with `sbatch pull_bencher_singularity.sbatch`.
    - **`bo-arena`**: build with `sbatch build_bo_arena_singularity.sbatch`. Note that this needs `--fakeroot`; on clusters without such access, build the image elsewhere and copy it in.

   The exact images used for running the experiments in the leaderboard can be found here.
3. Run experiments with `submit.py`. Commands used to run the experiments for the leaderboard can be found below. Note that `submit.py` performs operations such as setting time and memory limits, and deciding whether to run on CPU or GPU depending on the algorithm/task being run, so you may wish to edit this depending on your specific cluster setup. The main code for actually running the experiments, which is executed via `submit.py`, can be found in `scripts/experiment_runner.py`. The core logic is very similar to the code snippet provided in our paper, with additional code added for performing tasks such as (optionally) initialising a WandB run, setting up a logger, ensuring the working tree is clean for reproducibility, and saving results.
4. (Optional, if using WandB) - Sync WandB results after experiments finish using `bash submit.sh sync_wandb.sbatch`. Experiments log to WandB in offline mode to avoid API rate limits with many concurrent jobs.

Note that if you wish to run experiments without using the SLURM scripts, or building the Singularity image for `bo-arena`, you can do so by creating a virtual environment with the dependencies specified in `uv.lock` (which were used for running experiments for the leaderboard), and running the code in `scripts/experiment_runner.py` directly, ensuring you have a `bencher` server running and accessible at `localhost:50051`. The correct arguments to provide to `experiment_runner.py` can be inferred from the "Experiment Commands" section below.

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

This repository was also used to generate the results on the synthetic problems in Figure 4 of our paper. The commands used to obtain these results were as follows:

```
python slurm/submit.py \
--problem ackley_dim_100,michalewicz_dim_100,styblinski_tang_dim_100 \
--agent vanilla_bo,linear_bo,random,turbo \
--seed-range 0-19
```

and the results for `bounce` were obtained with:

```
python slurm/submit.py \
--problem ackley_dim_100,michalewicz_dim_100,styblinski_tang_dim_100 \
--agent bounce \
--n-init 5 \
--seed-range 0-19
```
