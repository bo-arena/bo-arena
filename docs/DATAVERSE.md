# bo-arena results

We store results in [our dataset on the Harvard dataverse](https://doi.org/10.7910/DVN/MNGMHT), so that you don't need to run benchmark problems yourself. The following files are present:

`results.csv` - one row per function evaluation:
- `problem` - Problem name; joins to `problems.csv`.
- `agent` - BO agent that produced the run.
- `seed` - Random seed identifying a single optimisation run.
- `iteration` - 0-based index of the function evaluation within the run.
- `y` - Observed black-box value (all problems are to be *maximised*).

`problems.csv` - one row per problem:
- `problem` - Problem name (primary key).
- `input_dim` - Dimensionality of the problem input space.
- `n_initial_points` - Number of initial design points evaluated before the BO loop.
- `n_total_evaluations` - Total number of function evaluations in a run.
- `stochastic` - Whether the problem is stochastic, i.e. repeated evaluations at the same input can return different `y` values.

### Plotting

Please refer to `notebooks/plotting.py` for information on how to generate the plots used as the leaderboard on our website.

### Semantics

- A run is identified by `(problem, agent, seed)`.
- `iteration` is the 0-based evaluation index within a run. The first `n_initial_points` evaluations (see `problems.csv`) form the initial design; the remainder are BO iterations. Note that the `bounce` agent is an exception - it samples 5 initial points for consistency with the original bounce paper.
- `y` is the raw observed value, not the cumulative best. Compute best-so-far (e.g. `numpy.maximum.accumulate`) downstream if needed.
- Most runs have 1000 evaluations; `saasbo` runs have 500 due to the algorithm's computational expense.
- `saasbo` was not run on `mujoco-humanoid` due to its computational expense.
- Two seeds for the `ibnn_depth_3` agent on `mopta08` are missing due to model-fitting failures on these runs.
