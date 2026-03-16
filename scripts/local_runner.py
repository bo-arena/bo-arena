"""Script for testing out agents locally, without full WandB and logging set up."""

import argparse
import logging

from acqf_opt import get_opt_kwargs

from bo_arena.agents import AGENTS
from bo_arena.arena import Arena
from bo_arena.benchmarks import BencherBenchmark

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
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

    return parser.parse_args()


def main() -> None:
    """Run the benchmark."""
    args = parse_args()

    logger.info(f"Starting run with Benchmark: {args.benchmark}, Agent: {args.agent}")

    # Initialize Benchmark
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

    # Initialize Agent
    agent_class = AGENTS[args.agent]
    agent = agent_class(
        input_dim=input_dim,
        opt_kwargs=opt_kwargs,
        n_total_evaluations=n_tot,
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
        wandb_run=None,
    )

    bo_arena.run()


if __name__ == "__main__":
    main()
