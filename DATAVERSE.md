# bo-arena benchmark results

We store results on the Harvard dataverse, so that you don't need to run benchmark problems yourself. The following files are present:

`results.csv` - one row per function evaluation:
- `problem` - Benchmark name; joins to benchmarks.csv.
- `agent` - BO agent that produced the run.
- `seed` - Random seed identifying a single optimisation run.
- `iteration` - 0-based index of the function evaluation within the run.
- `y` - Observed black-box value.

`benchmarks.csv` - one row per benchmark:
- `problem` - Benchmark name (primary key).
- `input_dim` - Dimensionality of the benchmark input space.
- `n_initial_points` - Initial design points evaluated before the BO loop.
- `n_total_evaluations` - Total function evaluations in a run.
- `stochastic` - Whether the benchmark is stochastic, i.e. repeated evaluations at the same input can return different y values.

### Semantics

- A run is identified by `(problem, agent, seed)`.
- `iteration` is the 0-based evaluation index within a run. The first `n_initial_points` evaluations (see `benchmarks.csv`) form the initial design; the remainder are BO iterations.
- `y` is the raw observed value, not cumulative best. Compute best-so-far (e.g. `numpy.maximum.accumulate`) downstream if needed.
- Most runs have 1000 evaluations; `saasbo` runs have 500.
