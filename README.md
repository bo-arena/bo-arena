# bo-arena
Arena for simple, reliable, and efficient benchmarking of Bayesian optimisation algorithms.

### Contributing

Please refer to the [contributing guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

### Running Experiments with Bencher on a SLURM Cluster
The `slurm` directory contains SLURM batch scripts for building Singularity containers and running `Bencher` and `bo-arena` experiments on a SLURM cluster. These scripts are designed to be run on the cluster and may require modifications to paths, resource specifications, and other parameters to fit your specific cluster setup. At a high level, there are two containers that need to be built and run:
1. `Bencher` container: This container runs the `Bencher` server, which is required for running `bo-arena` experiments. It should be started first and kept running while `bo-arena` experiments are executed. The `bo-arena` experiment containers will connect to the `Bencher` server to run experiments, and it communicates its availability by writing its IP address to `slurm/bencher_host.txt`. This means that `bo-arena` experiments can be run on separate nodes, and still connect to the server. When the `Bencher` server is no longer available (e.g., when the job finishes or is killed), it writes `OFFLINE` to `slurm/bencher_host.txt` to signal that experiment containers should not attempt to connect to it.
2. `bo-arena` experiment container: This container is used to run `bo-arena` experiments. It reads the `Bencher` server IP address from `slurm/bencher_host.txt` to connect to the server to run the benchmarks during experiments. There are separate scripts for running `bo-arena` on CPU or GPU, so make sure to use the appropriate one based on your needs.

In order to run experiments, follow these steps:
1. Build the Singularity images for `Bencher` and `bo-arena` using the provided SLURM batch scripts (`build_bencher_singularity.sbatch` and `build_cpu_bo_arena_singularity.sbatch` or `build_gpu_bo_arena_singularity.sbatch` depending on whether you want to run `bo-arena` on CPU or GPU). The images will be written to the `images` directory.
2. Start the `Bencher` container using the `run_bencher_singularity.sbatch` script. This will run the `Bencher` server on a node in the cluster and write its IP address to `slurm/bencher_host.txt` so that bo-arena experiment containers can connect to it.
3. Run `bo-arena` experiments using the `run_cpu_bo_arena_singularity.sbatch` or `run_gpu_bo_arena_singularity.sbatch` scripts. These scripts will read the `Bencher` server IP address from `slurm/bencher_host.txt` and connect to it to run benchmarks during experiments. Make sure to specify the desired benchmark and agent when submitting these jobs.
