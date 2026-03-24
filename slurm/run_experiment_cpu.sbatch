#!/bin/bash
#SBATCH --job-name=run_single_agent
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --array=0-9

# Override --array to run different seed ranges, e.g.:
#   bash slurm/submit.sh run_experiment_cpu.sbatch --export=ALL,BENCHMARK=lasso-dna,AGENT=vanilla_bo,ACQF_OPT_STRATEGY=default --array=10-19

# Required environment variables (set via submit.sh --export):
#   BENCHMARK          - benchmark name (e.g., "lasso-dna")
#   AGENT              - agent name (e.g., "vanilla_bo")
#   ACQF_OPT_STRATEGY  - acquisition function optimization strategy (e.g., "default")

source "$(pwd)/slurm/preamble.sh"

if [ -z "$BENCHMARK" ] || [ -z "$AGENT" ] || [ -z "$ACQF_OPT_STRATEGY" ]; then
    echo "Error: BENCHMARK, AGENT, and ACQF_OPT_STRATEGY must be set."
    echo "Usage: bash slurm/submit.sh run_experiment_cpu.sbatch --export=ALL,BENCHMARK=lasso-dna,AGENT=vanilla_bo,ACQF_OPT_STRATEGY=default"
    exit 1
fi

REPO=$(pwd)

# Start bencher as a background sidecar
singularity run images/bencher.sif &
BENCHER_PID=$!
trap "kill $BENCHER_PID" EXIT TERM

# Wait for Bencher services to be ready
echo "Starting up bencher server"
for port in 50051 50053 50054 50055 50056 50057 50058 50059 50060; do
    until nc -z 127.0.0.1 $port; do sleep 1; done
done
echo "Bencher server started"

# Run the experiment (SLURM_ARRAY_TASK_ID indexes into EXPERIMENT_SEEDS)
echo "Running: benchmark=$BENCHMARK, agent=$AGENT, acqf_opt_strategy=$ACQF_OPT_STRATEGY, seed_idx=$SLURM_ARRAY_TASK_ID"
singularity exec \
    --bind "$REPO:/workspace/bo-arena" \
    --env TMPDIR=/tmp \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    images/bo_arena.sif \
    bash -c "
        cd /workspace/bo-arena
        source /opt/venv/bin/activate
        python scripts/experiment_runner.py \
            --benchmark $BENCHMARK \
            --agent $AGENT \
            --acqf-opt-strategy $ACQF_OPT_STRATEGY \
            --seed-idx $SLURM_ARRAY_TASK_ID \
            --hostname 127.0.0.1 \
            --port 50051 \
            --device cpu \
            --dtype float64
    "
