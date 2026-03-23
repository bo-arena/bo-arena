#!/bin/bash
#SBATCH --job-name=build_bo_arena_singularity
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=01:00:00

# SLURM copies this script to /var/spool/slurm/... before running it, so
# dirname "$0" doesn't point to the repo. Use pwd instead (set via --chdir).
CONFIG="$(pwd)/slurm/config.sh"
if [ -f "$CONFIG" ]; then
    set -a; source "$CONFIG"; set +a
fi
# Batch jobs don't source login scripts, so `module` is unavailable by default.
# MODULES_INIT points to the module system init script that defines it.
if [ -n "$MODULES_INIT" ]; then
    source "$MODULES_INIT"
fi
# Load Singularity/Apptainer if a setup command is configured
if [ -n "$SINGULARITY_SETUP_CMD" ]; then
    eval "$SINGULARITY_SETUP_CMD"
fi

# Set cache and tmp directories for Singularity
export SINGULARITY_CACHEDIR=$SCRATCH
export SINGULARITY_TMPDIR=$SCRATCH
export APPTAINER_CACHEDIR=$SCRATCH
export APPTAINER_TMPDIR=$SCRATCH

# Build the bo-arena Singularity container
mkdir -p images
REPO=$(pwd)
singularity build --bind "$REPO:/workspace/bo-arena" --fakeroot images/bo_arena.sif slurm/bo_arena.def
