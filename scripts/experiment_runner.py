"""Main entry point for running BO arena experiments."""

import argparse
import json
import logging
import subprocess
from pathlib import Path

import botorch
import cpuinfo
import gpytorch
import numpy as np
import wandb

from bo_arena.agents import AGENTS
from bo_arena.arena import Arena
from bo_arena.benchmarks import BencherBenchmark

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

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

    agent_class = AGENTS[args.agent]
    agent = agent_class(
        input_dim=input_dim,
        acqf_opt_strategy=args.acqf_opt_strategy,
        n_total_evaluations=n_tot,
    )

    # Build experiment directory and file paths
    # Structure: results/{benchmark}/{agent}/acqf_opt_{strategy}/
    problem_name = args.benchmark
    agent_name = str(agent)
    acqf_opt_dir = f"acqf_opt_{args.acqf_opt_strategy}"
    experiment_dir = RESULTS_DIR / problem_name / agent_name / acqf_opt_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    results_file = experiment_dir / f"seed_{args.seed}.npz"
    log_file = experiment_dir / f"seed_{args.seed}.log"

    # Check for existing results/logs to avoid overwriting previous runs
    if results_file.exists() or log_file.exists():
        msg = (
            f"Results or log file already exists for seed {args.seed} in "
            f"{experiment_dir}. Refusing to overwrite."
        )
        raise FileExistsError(msg)

    # Build config and write to experiment directory (shared across seeds)
    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT_DIR,
    ).stdout.strip()

    config = {
        "benchmark": problem_name,
        "agent": agent_name,
        "acqf_opt_strategy": args.acqf_opt_strategy,
        "n_initial_points": n_init,
        "n_total_evaluations": n_tot,
        "batch_size": args.batch_size,
        "git_commit": git_hash or "unknown",
        "botorch_version": botorch.__version__,
        "gpytorch_version": gpytorch.__version__,
        "cpu": cpuinfo.get_cpu_info()["brand_raw"],
    }

    config_file = experiment_dir / "config.json"
    try:
        with config_file.open("x", encoding="utf-8") as f:
            f.write(json.dumps(config, indent=2) + "\n")
    except FileExistsError:
        existing_config = json.loads(config_file.read_text())
        if existing_config != config:
            msg = (
                f"Existing config.json in {experiment_dir} does not match current "
                f"config. This likely means experiment parameters have changed.\n"
                f"Existing: {existing_config}\n"
                f"Current: {config}"
            )
            raise RuntimeError(msg) from None

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    wandb_group = f"{problem_name}/{agent_name}/{acqf_opt_dir}"
    logger.info(f"Experiment: {wandb_group}, seed={args.seed}")
    logger.info(f"Config: {config}")

    # Initialize WandB
    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            entity="bo-arena",
            project="bo-arena",
            name=f"seed_{args.seed}",
            group=wandb_group,
            config={**config, "seed": args.seed},
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
