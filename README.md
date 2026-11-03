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

### Running Experiments on a Cluster with SLURM

Please refer to our more in-depth [usage docs](docs/USAGE.md) for details on how to run experiments on a cluster with SLURM with the provided scripts. These docs also contain the commands used to produce the `bo-arena` results.

> **⚠️ Warning**
>
> Running experiments on a cluster via `run_experiment.sbatch` creates temporary socket files for communication with the `bencher` server. Please refer to the [usage docs](docs/USAGE.md) for information on how to ensure that these files get cleaned up on your cluster.

### Obtaining the Dataset from the Harvard Dataverse

Results used to create the leaderboard are available on the Harvard Dataverse. For details on how to download the results and produce plots with them, please refer to our [dataverse docs](docs/DATAVERSE.md).
