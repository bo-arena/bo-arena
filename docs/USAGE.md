### Running Experiments with bencher on a SLURM Cluster

The `slurm` directory contains SLURM batch scripts for pulling Singularity images and running `bencher` and `bo-arena` experiments on a SLURM cluster. These scripts may require modifications to paths, resource specifications, and other parameters to fit your specific cluster setup.

Each experiment job is self-contained: it starts a `bencher` server as a sidecar process, runs the experiment, then shuts the server down on exit. This means no shared `bencher` server is needed, jobs scale independently, and there is no coordination between jobs. Each job gives its sidecar a private set of Unix domain sockets under `$TMPDIR/bencher-$SLURM_JOB_ID`, so jobs co-scheduled on the same node cannot reach each other's server. There are two containers involved per job:

1. **`bencher` container**: Runs the `bencher` server on the job's own sockets. Started as a background sidecar at the beginning of each experiment job and stopped automatically when the job exits.
2. **`bo-arena` experiment container**: Runs the `bo-arena` experiment, connecting to the `bencher` server over that job's router socket.

We use Unix sockets rather than fixed TCP ports to prevent different SLURM jobs from using each other's sidecars. TCP ports live in the node's network namespace, so jobs on the same node bind to the same TCP ports (50051, 50053-50060). In contrast, sockets are addressed by filesystem path, so putting the job id in the path means different jobs can't share each other's sidecar, avoiding this problem.

> **⚠️ Warning**
>
> Important - The run_experiment.sbatch script creates the directory $TMPDIR/bencher-$SLURM_JOB_ID (falling back to /tmp/bencher-$SLURM_JOB_ID if $TMPDIR is not set). It holds the socket files for communicating with bencher and is also mounted as the `bencher` container's /tmp, so it can contain bencher's temporary files too. The script generally removes this directory once it has finished running, but in some cases (e.g. SIGKILL) it will not. Ideally $TMPDIR should be a job-specific temporary directory, so it will get deleted once the job finishes running in any case (on many clusters this is the case). If not, be careful, as $TMPDIR could get cluttered with these directories.

In order to run experiments, follow these steps:
1. Copy `slurm/config.sh.example` to `slurm/config.sh` and fill in your
cluster-specific paths, partition names and WandB credentials (if running with WandB,
as is the default behaviour). This file is gitignored.
2. Obtain the Singularity images, which are expected to be in an `images/` directory.
   Links to the exact images used for running the experiments in the latest version of the leaderboard can be found in [CHANGELOG.md](CHANGELOG.md). Note that if you want to build a `bo-arena` image yourself, for instance, because you added a new dependency or need to update a dependency, then the instructions can be found at the top of `Dockerfile` (though you may not wish to push the resulting image to Docker Hub).
3. Run experiments with `submit.py`. Commands used to run the experiments for the leaderboard can be found in [CHANGELOG.md](CHANGELOG.md) (and the commands used for the original paper are in [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md)). Note that `submit.py` performs operations such as setting time and memory limits, and deciding whether to run on CPU or GPU depending on the algorithm/task being run, so you may wish to edit this depending on your specific cluster setup. The main code for actually running the experiments, which is executed via `submit.py`, can be found in `scripts/experiment_runner.py`. The core logic is very similar to the code snippet provided in our paper, with additional code added for performing tasks such as (optionally) initialising a WandB run, setting up a logger, ensuring the working tree is clean for reproducibility, and saving results.
4. (Optional, if using WandB) - Sync WandB results after experiments finish using `bash slurm/submit.sh sync_wandb.sbatch`. Experiments log to WandB in offline mode to avoid API rate limits with many concurrent jobs.

Note that if you wish to run experiments without using the SLURM scripts, or pulling the Singularity image for `bo-arena`, you can do so by creating a virtual environment with the dependencies specified in `uv.lock` (which were used for running experiments for the leaderboard), and running the code in `scripts/experiment_runner.py` directly, ensuring you have a `bencher` server running and accessible at `localhost:50051`. The correct arguments to provide to `experiment_runner.py` can be inferred from tracing the code run via the commands in [CHANGELOG.md](CHANGELOG.md).
