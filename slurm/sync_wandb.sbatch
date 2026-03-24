#!/bin/bash
#SBATCH --job-name=sync_wandb
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --time=24:00:00

# Syncs all offline WandB runs to the server.
#
# Usage:
#   bash slurm/submit.sh sync_wandb.sbatch

source "$(pwd)/slurm/preamble.sh"

REPO=$(pwd)

singularity exec \
    --bind "$REPO:/workspace/bo-arena" \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    images/bo_arena.sif \
    bash -c "
        cd /workspace/bo-arena
        source /opt/venv/bin/activate
        python -m wandb sync --sync-all wandb/
    "
