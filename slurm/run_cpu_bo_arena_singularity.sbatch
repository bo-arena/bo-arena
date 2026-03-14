#!/bin/bash
#SBATCH --job-name=run_cpu_bo_arena_singularity
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=06:00:00

REPO=$(pwd)
BENCHER_HOST_FILE=$REPO/slurm/bencher_host.txt
if [ ! -f "$BENCHER_HOST_FILE" ] || [ "$(cat $BENCHER_HOST_FILE)" = "OFFLINE" ]; then
    echo "Bencher server is offline. Start run_bencher_singularity.sbatch first."
    exit 1
fi
BENCHER_HOST=$(cat $BENCHER_HOST_FILE)

BENCHMARK="lasso-dna"
AGENT="vanilla_bo"
SEED=42

# Run the bo-arena Singularity container
singularity exec \
    --bind "$REPO:/workspace/bo-arena" \
    --env TMPDIR=/tmp \
    images/cpu_bo_arena.sif \
    bash -c "
        cd /workspace/bo-arena
        source .venv-cpu/bin/activate
        python scripts/local_runner.py \
            --benchmark $BENCHMARK \
            --agent $AGENT \
            --seed $SEED \
            --hostname $BENCHER_HOST \
            --port 50051
    "
