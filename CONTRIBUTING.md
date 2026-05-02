# Contributing Guidelines

Thanks for your interest in contributing to `bo-arena`! We welcome contributions from the community to help improve the project. Please follow the guidelines below to ensure a smooth contribution process.

## How to Contribute

1. **Fork the Repository**: Start by forking the `bo-arena` repository on GitHub to your own account.
2. **Clone Your Fork**: Clone your forked repository to your local machine using:
   ```
   git clone <your-fork-url>
   ```
3. **Install Dependencies**: Navigate to the project directory and install the required dependencies. We use [uv](https://docs.astral.sh/uv/) for managing dependencies, and highly recommend that you do too. To get started quickly, after installing `uv`, you can simply run:
    ```
    uv sync --all-extras
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
5. **Run Tests**: Before committing your changes, run the unittests to ensure everything is working correctly:
   ```
   uv run pytest -m "not regression" tests
   ```
6. **Commit Changes**: Commit your changes with a clear and descriptive commit message.
7. **Push Changes**: Push your changes to your forked repository:
   ```
   git push origin <new-feature-branch>
   ```
8. **Create a Pull Request**: Open a pull request in the original `bo-arena` repository.

## Notebooks

We use `jupytext` to manage Jupyer notebooks, storing them as `.py` files for better version control. In  order to convert between `.ipynb` and `.py` formats, you can use the following commands:

- To convert a `.ipynb` file to a `.py` file:
  ```
  jupytext --to py:percent <notebook>.ipynb
  ```

- To convert a `.py` file back to a `.ipynb` file:
  ```
  jupytext --to ipynb <notebook>.py
  ```

Note that you can prepend `uv run` to these commands if you have `jupytext` installed in your `uv`-managed virtual environment.


## Regression Tests

We have a regression test that compares the performance of registered agents against checked-in performance baselines. This is currently done on a 300D embedding of the Hartmann6D benchmark. This test is designed to always pass, and its primary purpose is to generate a report showing whether agent performance has changed relative to the baselines stored in `tests/regression_baselines/`. This helps us to monitor for any unintended performance regressions in agents over time, which may primarily happen due to:
1. Changes in this codebase affecting agent performance, or
2. Changes in upstream dependencies affecting agent performance (e.g. BoTorch, GPyTorch, etc.).

When adding an agent, you should generate baseline results for it by running:

```
uv run scripts/generate_regression_baselines.py --agents <agent-name>
```

This runs the agent on the synthetic benchmark and saves the resulting performance baseline in `tests/regression_baselines/`. You can then commit this baseline file to the repository so that it can be used for future regression testing. Please refer to `scripts/generate_regression_baselines.py` and `tests/test_regression.py` for more details on how the regression test works and how to generate baselines.
