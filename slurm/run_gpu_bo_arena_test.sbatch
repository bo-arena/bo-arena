#!/bin/bash
#SBATCH --job-name=run_gpu_bo_arena_singularity
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
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
    --nv \
    --env TMPDIR=/tmp \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env PYTHONPATH=/workspace/bo-arena \
    images/bo_arena.sif \
    bash -c "
        cd /workspace/bo-arena
        /opt/venv/bin/python scripts/local_runner.py \
            --benchmark $BENCHMARK \
            --agent $AGENT \
            --seed $SEED \
            --address 127.0.0.1 \
            --port 50051 \
            --device cuda \
            --dtype float64
    "
