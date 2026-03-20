#!/bin/bash
#SBATCH --job-name=sync_wandb
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --time=24:00:00

REPO=$(pwd)

# Load Singularity/Apptainer if a setup command is configured
if [ -n "$SINGULARITY_SETUP_CMD" ]; then
    eval "$SINGULARITY_SETUP_CMD"
fi

WANDB_DIR="$REPO/wandb"
SYNCED_DIR="$WANDB_DIR/synced"

if [ ! -d "$WANDB_DIR" ]; then
    echo "No wandb directory found at $WANDB_DIR"
    exit 1
fi

mkdir -p "$SYNCED_DIR"

RUNS=("$WANDB_DIR"/offline-run-*)
echo "Found ${#RUNS[@]} offline run(s) to sync"

SYNCED=0
FAILED=0

for run_dir in "${RUNS[@]}"; do
    [ -d "$run_dir" ] || continue
    echo "Syncing: $(basename "$run_dir")"

    singularity exec \
        --bind "$REPO:/workspace/bo-arena" \
        --env WANDB_API_KEY="$WANDB_API_KEY" \
        images/bo_arena.sif \
        bash -c "
            cd /workspace/bo-arena
            source .venv/bin/activate
            wandb sync --mark-synced '$run_dir' 2>&1
        "

    if [ $? -eq 0 ]; then
        mv "$run_dir" "$SYNCED_DIR/"
        SYNCED=$((SYNCED + 1))
        echo "  OK — moved to synced/"
    else
        FAILED=$((FAILED + 1))
        echo "  FAILED — left in place for retry"
    fi

    # Pause between syncs to avoid hitting WandB API rate limits
    sleep 2
done

echo "Done. Synced: $SYNCED, Failed: $FAILED"
