# bo-arena
Arena for simple, reliable, and efficient benchmarking of Bayesian optimisation algorithms.

### Contributing

Please refer to the [contributing guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

### Running Experiments with Bencher on a SLURM Cluster
The `slurm` directory contains SLURM batch scripts for building Singularity containers and running `Bencher` and `bo-arena` experiments on a SLURM cluster. These scripts are designed to be run on the cluster and may require modifications to paths, resource specifications, and other parameters to fit your specific cluster setup. At a high level, there are two containers that need to be built and run:
1. `Bencher` container: This container runs the `Bencher` server, which is required for running `bo-arena` experiments. It should be started first and kept running while `bo-arena` experiments are executed. The `bo-arena` experiment containers will connect to the `Bencher` server to run experiments, and it communicates its availability by writing its `IP:PORT` to `slurm/bencher_host.txt`. This means that `bo-arena` experiments can be run on separate nodes, and still connect to the server. When the `Bencher` server is no longer available (e.g., when the job finishes or is killed), it writes `OFFLINE` to `slurm/bencher_host.txt` to signal that experiment containers should not attempt to connect to it.
2. `bo-arena` experiment container: This container is used to run `bo-arena` experiments. It reads the `Bencher` server `IP:PORT` from `slurm/bencher_host.txt` to connect to the server to run the benchmarks during experiments. There are separate scripts for running `bo-arena` on CPU or GPU, so make sure to use the appropriate one based on your needs.

In order to run experiments, follow these steps:
1. Copy `slurm/config.sh.example` to `slurm/config.sh` and fill in your cluster-specific paths, partition names, and Bencher port. This file is gitignored and must be created by each user. If multiple users share the same cluster nodes, use different `BENCHER_PORT` values to avoid collisions.
2. Build the Singularity images using `slurm/submit.sh` (e.g. `slurm/submit.sh build_bencher_singularity.sbatch`). One person should build the images and share them with collaborators (e.g. via `cp` on the cluster) to ensure identical environments.
3. Start the `Bencher` container using `slurm/submit.sh run_bencher_singularity.sbatch`. This writes the server's `IP:PORT` to `slurm/bencher_host.txt` so experiment jobs can connect to it.
4. Run `bo-arena` experiments using `slurm/submit.sh run_cpu_bo_arena_singularity.sbatch` or `slurm/submit.sh run_gpu_bo_arena_singularity.sbatch`.

All jobs should be submitted via `slurm/submit.sh` rather than `sbatch` directly — it sources `slurm/config.sh` and passes the cluster-specific `--chdir`, `--output`, and `--partition` flags to `sbatch`.
