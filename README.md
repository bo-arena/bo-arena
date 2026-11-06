# bo-arena

A Python package for benchmarking high-dimensional Bayesian optimisation (HDBO) algorithms, with verified implementations of the current state of the art.

Most HDBO papers re-implement their baselines, and small differences between implementations can lead to very different conclusions. `bo-arena` contains a single implementation of each of the main HDBO algorithms (Vanilla BO, `TuRBO`, `SAASBO`, `Bounce`, Linear BO and I-BNNs), verified against the results in the original papers. All algorithms can be run on the 40+ continuous problems in [bencher](https://github.com/lpapenme/bencher) (e.g. `Rover`, `MOPTA08`, `Lasso-DNA`, `SVM`, MuJoCo `Ant` and `Humanoid`), or on your own problem by subclassing `Problem` and implementing `evaluate_points`. New algorithms can be added by implementing the `AskTellAgent` interface, and can then be run on the same range of problems using just a single command. All runs are seeded, so results are reproducible.

The results behind the leaderboard on our [website](https://bo-arena.github.io) are [publicly available](https://doi.org/10.7910/DVN/MNGMHT), so new methods can be compared against these results without re-running any baselines. More details can be found in our paper, `BO-Arena: An Evolving Benchmark for High-Dimensional Bayesian Optimisation` (NeurIPS 2026, Evaluations & Datasets track; see the [citation](#citation) below). The code for the ablations in the paper is in a [separate repository](https://github.com/bo-arena/bo-arena-neurips2026).

### Contributing

Please refer to the [contributing guidelines](CONTRIBUTING.md) for details on how to contribute to this project, including how to add your algorithm to the leaderboard.

### Quickstart

1. Install dependencies using `uv` (see the "Install Dependencies" section of the [contributing guidelines](CONTRIBUTING.md)).
2. Start up the `bencher` server. First pull the Docker image:

    ```
    docker pull --platform linux/amd64 tchristie/bencher@sha256:a7ca83f6d0b3810ed8dc03bb210aa35c46e65d5d19e4371aa7dbf038be3b6467
    ```

    Then start the server:

    ```
    docker run --platform linux/amd64 -p 50051:50051 --restart always -d tchristie/bencher@sha256:a7ca83f6d0b3810ed8dc03bb210aa35c46e65d5d19e4371aa7dbf038be3b6467
    ```
    This image was built from the `bencher` repository at commit `ba9ada07a7f3eaa5efd691200ed4866ff4ca6e1b`. If you want to build your own image, see [the bencher README](https://github.com/lpapenme/bencher).
3. Run an algorithm on a problem:

    ```python
    from bo_arena.agents import VanillaBO
    from bo_arena.arena import Arena
    from bo_arena.problems import BencherProblem

    problem = BencherProblem(name="svm")
    agent = VanillaBO(input_dim=problem.metadata.input_dim)
    arena = Arena(
        problem=problem,
        agent=agent,
        seed=43,
        batch_size=1,
        n_initial_points=30,
        n_total_evaluations=100,
    )
    result = arena.run()  # evaluated points, objective values and timings
    ```

    The same example is in the `quickstart` notebook in `notebooks/`. For converting between `.py` and `.ipynb`, see the "Notebooks" section of the [contributing guidelines](CONTRIBUTING.md).

### Running Experiments on a Cluster with SLURM

Please refer to our more in-depth [usage docs](docs/USAGE.md) for details on how to run experiments on a SLURM-based cluster with the provided scripts. These docs also contain the commands used to produce the `bo-arena` results.

> **⚠️ Warning**
>
> Running experiments on a cluster via `run_experiment.sbatch` creates temporary socket files for communication with the `bencher` server. Please refer to the [usage docs](docs/USAGE.md) for information on how to ensure that these files get cleaned up on your cluster.

### Obtaining the Dataset from the Harvard Dataverse

Results used to create the leaderboard are available on the Harvard Dataverse. For details on how to download the results and produce plots, please refer to our [dataverse docs](docs/DATAVERSE.md).

### Citation

If you use `bo-arena` in your work, please consider citing our paper:

```bibtex
@inproceedings{doumont2026boarena,
  title     = {{BO-Arena}: An Evolving Benchmark for High-Dimensional {B}ayesian Optimisation},
  author    = {Doumont, Colin and Christie, Thomas and Moss, Henry and Hennig, Philipp},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
