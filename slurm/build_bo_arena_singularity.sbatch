#!/bin/bash
#SBATCH --job-name=build_bo_arena_singularity
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=01:00:00

# Set cache and tmp directories for Singularity
export SINGULARITY_CACHEDIR=$SCRATCH
export SINGULARITY_TMPDIR=$SCRATCH
export APPTAINER_CACHEDIR=$SCRATCH
export APPTAINER_TMPDIR=$SCRATCH

# Build the bo-arena Singularity container
mkdir -p images
REPO=$(pwd)
singularity build --bind "$REPO:/workspace/bo-arena" --fakeroot images/bo_arena.sif slurm/bo_arena.def
