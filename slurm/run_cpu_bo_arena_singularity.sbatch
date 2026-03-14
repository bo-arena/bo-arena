#!/bin/bash
#SBATCH --job-name=run_cpu_bo_arena_singularity
#SBATCH --mem=16G
#SBATCH --partition=cpu-galvani
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH -o /mnt/lustre/work/hennig/hmx892/logs/%j_%x_%a.out
#SBATCH --chdir=/mnt/lustre/work/hennig/hmx892/repos/bo-arena

REPO=/mnt/lustre/work/hennig/hmx892/repos/bo-arena
BENCHER_HOST_FILE=/mnt/lustre/work/hennig/hmx892/repos/bo-arena/slurm/bencher_host.txt
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
    --bind /mnt/lustre/work/hennig/hmx892/repos/bo-arena:/mnt/lustre/work/hennig/hmx892/repos/bo-arena \
    --env TMPDIR=/tmp \
    images/cpu_bo_arena.sif \
    bash -c "
        cd $REPO
        source .venv-cpu/bin/activate
        python scripts/local_runner.py \
            --benchmark $BENCHMARK \
            --agent $AGENT \
            --seed $SEED \
            --hostname $BENCHER_HOST \
            --port 50051
    "
