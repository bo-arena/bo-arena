#!/bin/bash
#SBATCH --job-name=build_bencher_singularity
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=04:00:00

# Load Singularity/Apptainer if a setup command is configured
if [ -n "$SINGULARITY_SETUP_CMD" ]; then
    eval "$SINGULARITY_SETUP_CMD"
fi

# Set cache and tmp directories for Singularity
export SINGULARITY_CACHEDIR=$SCRATCH
export SINGULARITY_TMPDIR=$SCRATCH
export APPTAINER_CACHEDIR=$SCRATCH
export APPTAINER_TMPDIR=$SCRATCH

# Build the Singularity container
mkdir -p images
singularity build images/bencher.sif docker://gaunab/bencher:latest
