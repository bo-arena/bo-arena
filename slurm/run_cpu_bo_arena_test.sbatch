#!/bin/bash
#SBATCH --job-name=run_cpu_bo_arena_singularity
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=06:00:00

source "$(pwd)/slurm/preamble.sh"

REPO=$(pwd)

BENCHMARK="lasso-dna"
AGENT="vanilla_bo"
SEED=42

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

# Run the experiment
singularity exec \
    --bind "$REPO:/workspace/bo-arena" \
    --env TMPDIR=/tmp \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    images/bo_arena.sif \
    bash -c "
        cd /workspace/bo-arena
        source /opt/venv/bin/activate
        python scripts/local_runner.py \
            --benchmark $BENCHMARK \
            --agent $AGENT \
            --seed $SEED \
            --hostname 127.0.0.1 \
            --port 50051 \
            --device cpu \
            --dtype float64
    "
