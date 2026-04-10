#!/bin/bash
#SBATCH --job-name=run_experiment
#SBATCH --nodes=1

# Unified experiment script for CPU and GPU jobs. All resource parameters
# (--mem, --time, --partition, --gres, --array) are set by submit.py at
# submission time, not hardcoded here.
#
# Required environment variables (set via submit.py --export):
#   BENCHMARK          - benchmark name (e.g., "lasso-dna")
#   AGENT              - agent name (e.g., "vanilla_bo")
#   ACQF_OPT_STRATEGY  - acquisition function optimization strategy (e.g., "default")
#   DEVICE             - "cpu" or "cuda"
#
# Optional environment variables:
#   N_EVALS            - total number of evaluations (overrides benchmark default)

source "$(pwd)/slurm/preamble.sh"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

if [ -z "$BENCHMARK" ] || [ -z "$AGENT" ] || [ -z "$ACQF_OPT_STRATEGY" ] || [ -z "$DEVICE" ]; then
    echo "Error: BENCHMARK, AGENT, ACQF_OPT_STRATEGY, and DEVICE must be set."
    echo "Use submit.py to launch experiments."
    exit 1
fi

REPO=$(pwd)

# Enable GPU passthrough into the container when running on CUDA
NV_FLAG=""
if [ "$DEVICE" = "cuda" ]; then
    NV_FLAG="--nv"
fi

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
echo "Running: benchmark=$BENCHMARK, agent=$AGENT, acqf_opt_strategy=$ACQF_OPT_STRATEGY, device=$DEVICE, seed_idx=$SLURM_ARRAY_TASK_ID"
singularity exec \
    --bind "$REPO:/workspace/bo-arena" \
    $NV_FLAG \
    --env TMPDIR=${TMPDIR:-/tmp} \
    --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env PYTHONPATH=/workspace/bo-arena \
    images/bo_arena.sif \
    bash -c "
        cd /workspace/bo-arena
        /opt/venv/bin/python scripts/experiment_runner.py \
            --benchmark $BENCHMARK \
            --agent $AGENT \
            --acqf-opt-strategy $ACQF_OPT_STRATEGY \
            --seed-idx $SLURM_ARRAY_TASK_ID \
            --address 127.0.0.1 \
            --port 50051 \
            --device $DEVICE \
            --dtype float64 \
            ${N_EVALS:+--n-evals $N_EVALS}
    "
