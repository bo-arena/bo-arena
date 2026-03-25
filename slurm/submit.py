"""Submit experiment grid to SLURM.

Computes the cross product of benchmarks, agents, and strategies, looks up
the right SLURM resources for each combination, and submits one job array
per combination (one array task per seed).

Usage:
    # Preview what would be submitted:
    python slurm/submit.py \
        --benchmark rover,svm \
        --agent vanilla_bo,saasbo \
        --acqf-opt-strategy default,raasp \
        --seed-range 0-9 \
        --dry-run

    # Actually submit:
    python slurm/submit.py \
        --benchmark rover,svm \
        --agent vanilla_bo,saasbo \
        --acqf-opt-strategy default,raasp \
        --seed-range 0-9
"""

import argparse
import itertools
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SBATCH_TEMPLATE = SCRIPT_DIR / "run_experiment.sbatch"

# ---------------------------------------------------------------------------
# Resource profiles keyed by agent name. Falls back to "default".
# Add new agents here as needed.
# ---------------------------------------------------------------------------
RESOURCE_PROFILES = {
    "saasbo": {
        "device": "cuda",
        "gres": "gpu:1",
        "mem": "32G",
        "time": "72:00:00",
    },
    "default": {
        "device": "cpu",
        "mem": "16G",
        "time": "36:00:00",
    }
}


def parse_config_sh() -> dict[str, str]:
    """Read key=value pairs from slurm/config.sh."""
    config_path = SCRIPT_DIR / "config.sh"
    if not config_path.exists():
        print(
            f"Error: {config_path} not found.\n"
            "Copy slurm/config.sh.example to slurm/config.sh and fill in your paths.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = {}
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip().strip('"').strip("'")

    required_keys = ["REPO", "LOGS", "CPU_PARTITION", "GPU_PARTITION"]
    missing = [k for k in required_keys if not config.get(k)]
    if missing:
        print(
            f"Error: config.sh is missing keys: {', '.join(missing)}.\n"
            "See slurm/config.sh.example for the expected format.",
            file=sys.stderr,
        )
        sys.exit(1)

    return config


def get_profile(agent: str) -> dict:
    """Return the resource profile for the given agent, or the default."""
    return RESOURCE_PROFILES.get(agent, RESOURCE_PROFILES["default"])


def build_sbatch_cmd(
    benchmark: str,
    agent: str,
    acqf_opt_strategy: str,
    seed_range: str,
    config: dict[str, str],
) -> list[str]:
    """Build the sbatch command for a single experiment combination."""
    profile = get_profile(agent)
    partition_key = "GPU_PARTITION" if profile["device"] == "cuda" else "CPU_PARTITION"
    partition = config[partition_key]

    export_vars = ",".join([
        "ALL",
        f"BENCHMARK={benchmark}",
        f"AGENT={agent}",
        f"ACQF_OPT_STRATEGY={acqf_opt_strategy}",
        f"DEVICE={profile['device']}",
    ])

    cmd = [
        "sbatch",
        f"--chdir={config['REPO']}",
        f"--partition={partition}",
        f"--mem={profile['mem']}",
        f"--time={profile['time']}",
        f"--array={seed_range}",
        f"--export={export_vars}",
        "-o", f"{config['LOGS']}/%j_%x_%a.out",
    ]

    if "gres" in profile:
        cmd.append(f"--gres={profile['gres']}")

    cmd.append(str(SBATCH_TEMPLATE))
    return cmd


def main() -> None:
    """Parse arguments and submit experiment grid to SLURM."""
    parser = argparse.ArgumentParser(
        description="Submit experiment grid to SLURM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Comma-separated benchmark names (e.g., rover,svm,lasso-dna)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Comma-separated agent names (e.g., vanilla_bo,saasbo)",
    )
    parser.add_argument(
        "--acqf-opt-strategy",
        type=str,
        required=True,
        help="Comma-separated strategies (e.g., default,raasp)",
    )
    parser.add_argument(
        "--seed-range",
        type=str,
        default="0-9",
        help="SLURM array range for seeds (default: 0-9)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"\d+(-\d+)?(%\d+)?(,\d+(-\d+)?(%\d+)?)*", args.seed_range):
        parser.error(
            f"Invalid --seed-range format: '{args.seed_range}'. "
            "Expected a SLURM array spec like '0-9', '0,2,4', or '0-9%3'."
        )

    config = parse_config_sh()
    Path(config["LOGS"]).mkdir(parents=True, exist_ok=True)

    benchmarks = [s.strip() for s in args.benchmark.split(",")]
    agents = [s.strip() for s in args.agent.split(",")]
    strategies = [s.strip() for s in args.acqf_opt_strategy.split(",")]

    grid = list(itertools.product(benchmarks, agents, strategies))

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Submitting {len(grid)} job array(s) "
          f"(seeds {args.seed_range} each):\n")

    submitted = 0
    failed = 0

    for benchmark, agent, strategy in grid:
        cmd = build_sbatch_cmd(benchmark, agent, strategy, args.seed_range, config)
        profile = get_profile(agent)
        dev, mem = profile["device"], profile["mem"]
        label = f"  {benchmark} / {agent} / {strategy}  [{dev}, {mem}]"

        if args.dry_run:
            print(label)
            print(f"    {' '.join(cmd)}\n")
        else:
            print(label)
            result = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                print(f"    {result.stdout.strip()}")
                submitted += 1
            else:
                print(f"    FAILED: {result.stderr.strip()}")
                failed += 1

    if not args.dry_run:
        print(f"\nDone. Submitted: {submitted}, Failed: {failed}")


if __name__ == "__main__":
    main()
