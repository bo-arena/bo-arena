"""Main entry point for running BO arena experiments."""

import argparse
import logging
import subprocess
from pathlib import Path

import botorch
import gpytorch
import numpy as np
import wandb
from acqf_opt import get_opt_kwargs

from bo_arena.agents import AGENTS
from bo_arena.arena import Arena
from bo_arena.benchmarks import BencherBenchmark

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR = ROOT_DIR / "logs"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Bayesian Optimization Benchmark")

    parser.add_argument(
        "--benchmark",
        type=str,
        default="lasso-dna",
        help="Name of the benchmark to run (e.g., 'svm')",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="random",
        choices=list(AGENTS.keys()),
        help=f"Agent to use. Available: {', '.join(AGENTS.keys())}",
    )
    parser.add_argument(
        "--acqf-opt-strategy",
        type=str,
        default="default",
        choices=["default", "no_raasp", "raasp"],
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n-init",
        type=int,
        default=None,
        dest="n_initial_points",
        help="Number of initial points",
    )
    parser.add_argument(
        "--n-evals",
        type=int,
        default=None,
        dest="n_total_evaluations",
        help="Total number of evaluations",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--hostname",
        type=str,
        default="127.0.0.1",
        help="Hostname of the Bencher server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port of the Bencher server",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )

    return parser.parse_args()


def main() -> None:
    """Run the benchmark."""
    # Ensure working tree is clean for reproducibility
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT_DIR,
    ).stdout.strip()
    if git_status:
        msg = (
            "Uncommitted changes detected. Commit or stash them before running "
            "experiments to ensure reproducibility.\n"
            f"Dirty files:\n{git_status}"
        )
        raise RuntimeError(msg)

    args = parse_args()

    # Initialize Benchmark and Agent
    benchmark = BencherBenchmark(
        name=args.benchmark, hostname=args.hostname, port=args.port
    )
    metadata = benchmark.metadata
    input_dim = metadata.input_dim

    # Resolve n_init and n_tot
    n_init = (
        args.n_initial_points
        if args.n_initial_points is not None
        else metadata.default_n_initial_points
    )
    n_tot = (
        args.n_total_evaluations
        if args.n_total_evaluations is not None
        else metadata.default_n_total_evaluations
    )

    opt_kwargs = get_opt_kwargs(args.acqf_opt_strategy)

    agent_class = AGENTS[args.agent]
    agent = agent_class(
        input_dim=input_dim,
        opt_kwargs=opt_kwargs,
        n_total_evaluations=n_tot,
    )

    # Build experiment name and directory paths
    problem_name = args.benchmark
    agent_name = str(agent)
    experiment_name = f"{problem_name}_{agent_name}_n_init_{n_init}_n_tot_{n_tot}_bs_{args.batch_size}_seed_{args.seed}"  # noqa: E501
    results_dir = RESULTS_DIR / problem_name / agent_name
    log_dir = LOGS_DIR / problem_name / agent_name
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing results/logs to avoid overwriting previous runs
    results_file = results_dir / f"{experiment_name}.npz"
    log_file = log_dir / f"{experiment_name}.log"
    if results_file.exists() or log_file.exists():
        msg = "Experiment and/or log files already exist. Refusing to overwrite."
        raise FileExistsError(msg)

    config = {
        "benchmark": problem_name,
        "agent": agent_name,
        "seed": args.seed,
        "n_initial_points": n_init,
        "n_total_evaluations": n_tot,
        "batch_size": args.batch_size,
    }

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Config: {config}")
    logger.info(
        f"Versions: botorch={botorch.__version__}, gpytorch={gpytorch.__version__}"
    )
    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT_DIR,
    ).stdout.strip()
    logger.info(f"Git commit: {git_hash or 'unknown'}")

    # Initialize WandB
    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            entity="bo-arena",
            project="bo-arena",
            name=experiment_name,
            group=f"{problem_name}_{agent_name}",
            config=config,
        )

    # Run Arena
    bo_arena = Arena(
        benchmark=benchmark,
        agent=agent,
        seed=args.seed,
        n_initial_points=n_init,
        n_total_evaluations=n_tot,
        batch_size=args.batch_size,
        logger=logger,
        wandb_run=wandb_run,
    )
    queried_points = bo_arena.run()

    # Save results
    np.savez(results_file, X=queried_points.X, y=queried_points.y)
    logger.info(f"Results saved to {results_file}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
