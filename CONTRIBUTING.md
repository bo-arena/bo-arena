# Contributing Guidelines

Thanks for your interest in contributing to `bo-arena`! We welcome contributions from the community to help improve the project. For general software engineering instructions for contributing (setting up `pre-commit`, running notebooks etc.) please refer to [Software Engineering](#software-engineering). For instructions on contributing a new agent (and its corresponding data for the leaderboard), please refer to [Contributing a New Agent](#contributing-a-new-agent).

## Software Engineering

1. **Fork the Repository**: Start by forking the `bo-arena` repository on GitHub to your own account.
2. **Clone Your Fork**: Clone your forked repository to your local machine using:
   ```
   git clone <your-fork-url>
   ```
3. **Install Dependencies**: Navigate to the project directory and install the required dependencies. We use [uv](https://docs.astral.sh/uv/) for managing dependencies, and highly recommend that you do too. To get started quickly, after installing `uv`, you can simply run:
    ```
    uv sync --locked --all-groups
    ```
    and this will create a virtual environment and install all necessary packages for development.

3. **Set up pre-commit hooks**: We use pre-commit hooks to maintain code quality. To initialise pre-commit hooks, run the following command:
   ```
   uv run pre-commit install
   ```
3. **Create a Branch**: Create a new branch for your feature or bug fix:
   ```
   git checkout -b <new-feature-branch>
   ```
4. **Make Changes**: Implement your changes in the codebase. Please ensure that your code includes appropriate tests.
5. **Run Tests**: Before committing your changes, run the unit tests to ensure everything is working correctly:
   ```
   uv run pytest tests
   ```
6. **Commit Changes**: Commit your changes with a clear and descriptive commit message.
7. **Push Changes**: Push your changes to your forked repository:
   ```
   git push origin <new-feature-branch>
   ```
8. **Create a Pull Request**: Open a pull request in the original `bo-arena` repository.

### Notebooks

We use `jupytext` to manage Jupyter notebooks, storing them as `.py` files for better version control. In  order to convert between `.ipynb` and `.py` formats, you can use the following commands:

- To convert a `.ipynb` file to a `.py` file:
  ```
  jupytext --to py:percent <notebook>.ipynb
  ```

- To convert a `.py` file back to a `.ipynb` file:
  ```
  jupytext --to ipynb <notebook>.py
  ```

Note that you can prepend `uv run` to these commands if you have `jupytext` installed in your `uv`-managed virtual environment.

## Contributing a New Agent

We would be very happy to add new agents to the arena, and add the results obtained with new agents to our leaderboard. In order to contribute a new algorithm, we request that you first open an issue in the repository in order to document the necessary information.

### Prerequisites

We require the following prerequisites to be satisfied in order to add a new algorithm to the repository:

1. **Algorithm verification** - The performance of the implementation should be verified, preferably by running the algorithm on the problems on our leaderboard, and comparing to the performance of the algorithm as reported in its original paper. Merging the implementation into the codebase will also require the approval of two maintainers of `bo-arena`, who will carefully review the implementation in order to verify its correctness, and try to make the implementation as clean as possible, and in the style of other algorithms within the repository. It should also pass the default unit tests for all algorithms (in particular, ensuring determinism of runs with a fixed random seed). If the implementation is contributed by the original authors of the algorithm, an attribute of the `AskTellAgent` implementation will specify that the implementation has been officially verified. If the implementation has not been made by the original author, we request that the contributor contacts them for verification.

2. **Consistent implementation** - We require that implementations of new algorithms are consistent with the existing codebase. In particular, new algorithms must implement the `AskTellAgent` abstract base class, and adhere to the design principles laid out in [DESIGN.md](docs/DESIGN.md). We also request that new implementations minimise the addition of new dependencies, and build on the `BoTorch` and `GPyTorch` ecosystem where possible. This vastly simplifies maintenance of the library, and aids in the scientific integrity of conclusions made using `bo-arena`. This is because there is typically a myriad of implementation details under the hood of BO algorithms (e.g. the exact Gaussian process implementation, L-BFGS-B implementation etc.), which would benefit from being shared between implementations.

3. **State-of-the-art performance or novel insights** - We require that implementations demonstrate state-of-the-art performance on a core set of problems, or yield novel insights. Currently, we deem this core set of problems to be those used in our
leaderboard. However, we are open to engaging in discussions around this. As detailed in our paper, we believe that the question of which problems to benchmark on in order to draw reliable conclusions about the relative performance of algorithms remains open, and we are happy to alter the core problems we use in the future. At the same time, unless there is strong evidence that the problems should change, we intend on keeping them fairly constant, in order to avoid continually having to re-run existing algorithms on new problems.

   We recognise that defining "novel insights" is more nuanced, and we will likely have to handle the definition of this criterion on a case-by-case basis. A concrete example is the `LinearBO` agent - whilst one might argue that it does not "outperform" Vanilla BO, its use of a linear model can yield computational gains, and the method lead to surprising insights into what is necessary for achieving competitive performance in high-dimensional BO.

4. **Complete algorithm specification** - We require that the implementation of the BO algorithm as an `AskTellAgent` is fully self-contained where possible, and does not require the user to specify problem-dependent hyperparameters. We require this as our goal is to provide a codebase where practitioners and researchers can take any algorithm off the shelf and apply it to a new problem without having to select any hyperparameters.

5. **Maintenance commitment** - We require that contributors commit to assisting us in maintaining their implementation, and provide a means of contacting them to do so. We do not envisage this being a large commitment. For instance, we might ask that the original contributor provides approval on a pull request, should someone find a bug in their implementation. We might also request that the contributor commits to re-running experiments should bugs be found, or should new problems be added, particularly in the case of the implementation having a heavy computational requirement.

6. **Result generation** - We require contributors to generate results of new algorithms on the core set of problems used in the benchmark at the time. We provide further details on doing this below. We believe that the core set of problems will stay fairly constant over time, unless strong evidence appears which suggests they should be changed, so we hope that results generated by authors will be usable for an extended period of time. They will be made available to download from [our dataset on the Harvard Dataverse](https://doi.org/10.7910/DVN/MNGMHT), eliminating the need for future papers to re-run experiments on these problems.

### Result Generation

One of the aims of `bo-arena` is improved reproducibility of results. In order to improve reproducibility, we require the following:

1. Full specification of dependencies - In an ideal scenario, you would use the same Singularity image used by us in order to generate results, which would use the exact same dependencies if run via the scripts provided (see the instructions in the [README.md](README.md)). However, we accept that in some cases this will not be possible, for instance if your agent requires the addition of a new dependency, or for one of our existing dependencies to be bumped. In this case, we require that you provide, at minimum, the exact `uv.lock` file used for installing the dependencies used to generate results, and ideally the Singularity image used too.
2. A link to the hash used for running experiments (once your agent has been merged into `main`, see below).
3. Using the same experimental setup as us for experiments (e.g. random seeds etc.), which is guaranteed if you follow the instructions provided in the [README.md](README.md).

Final results should be generated *after* merging your PR into `bo-arena`, such that the results in the leaderboard correspond exactly to those generated by the implementation in our repository.

### Code Requirements

Besides adhering to the design principles laid out in [DESIGN.md](DESIGN.md), we require implementations to follow the following steps:

1. Add your agent to `bo_arena/agents/registry.py`, which will automatically make it get tested for determinism via the tests in `tests/test_seed_determinism.py`.
2. Make sure the `__str__` method returns an appropriate name for your agent, as this is used to form the directory name for storing generated results, as well as the name used in the dataset hosted on the Dataverse.
3. Add a resource profile in `slurm/submit.py` if your agent requires a GPU or more time/memory than the defaults.

When you open your PR, please also describe the hardware used to generate results (e.g. GPU model etc.).
