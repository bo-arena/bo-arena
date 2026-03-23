# bo-arena
Arena for simple, reliable, and efficient benchmarking of Bayesian optimisation algorithms.

### Contributing

Please refer to the [contributing guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

### Running Experiments with Bencher on a SLURM Cluster
The `slurm` directory contains SLURM batch scripts for building Singularity containers and running `Bencher` and `bo-arena` experiments on a SLURM cluster. These scripts are designed to be run on the cluster and may require modifications to paths, resource specifications, and other parameters to fit your specific cluster setup.

Each experiment job is self-contained: it starts a `Bencher` server as a sidecar process on `localhost`, runs the experiment, then shuts the server down on exit. This means no shared `Bencher` server is needed, jobs scale independently, and there is no coordination between jobs. Multiple jobs running concurrently on the same node are safe — the Linux kernel's `SO_REUSEPORT` allows multiple `Bencher` instances to bind to the same ports simultaneously and distributes requests across them. There are two containers involved per job:

1. **`Bencher` container**: Runs the `Bencher` server on `localhost`. Started as a background sidecar at the beginning of each experiment job and stopped automatically when the job exits.
2. **`bo-arena` experiment container**: Runs the `bo-arena` experiment, connecting to the `Bencher` server on `localhost:50051`. A single container is used for both CPU and GPU jobs.

In order to run experiments, follow these steps:
1. Copy `slurm/config.sh.example` to `slurm/config.sh` and fill in your cluster-specific paths and partition names. This file is gitignored and must be created by each user.
2. Build the Singularity images using `slurm/submit.sh` (e.g. `bash slurm/submit.sh build_bencher_singularity.sbatch` and `bash slurm/submit.sh build_bo_arena_singularity.sbatch`). One person should build the images and share them with collaborators (e.g. via `cp` on the cluster) to ensure identical environments.
3. Run experiments by specifying the benchmark, agent, and acquisition function optimization strategy:
   ```bash
   bash slurm/submit.sh run_experiment_cpu.sbatch \
       --export=ALL,BENCHMARK=lasso-dna,AGENT=vanilla_bo,ACQF_OPT_STRATEGY=default
   ```
   This submits a SLURM job array that runs the agent across the first 10 experiment seeds (indices 0–9). To run the second batch of seeds, override the array range:
   ```bash
   bash slurm/submit.sh run_experiment_cpu.sbatch \
       --export=ALL,BENCHMARK=lasso-dna,AGENT=vanilla_bo,ACQF_OPT_STRATEGY=default \
       --array=10-19
   ```
   Each job starts its own `Bencher` sidecar automatically — no need to start a separate `Bencher` job first.
4. Sync WandB results after experiments finish using `bash slurm/submit.sh sync_wandb.sbatch`. Experiments log to WandB in offline mode to avoid API rate limits with many concurrent jobs. The sync script uploads runs one at a time, moving successful uploads to `wandb/synced/`. Failed syncs are left in place so you can re-run the script to retry only those.

All jobs should be submitted via `slurm/submit.sh` rather than `sbatch` directly — it sources `slurm/config.sh` and passes the cluster-specific `--chdir`, `--output`, and `--partition` flags to `sbatch`.
