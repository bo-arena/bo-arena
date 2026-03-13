#!/bin/bash
#SBATCH --job-name=build_bencher_singularity
#SBATCH --mem=16G
#SBATCH --partition=cpu-galvani
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH -o /mnt/lustre/work/hennig/hmx892/logs/%j_%x_%a.out
#SBATCH --chdir=/mnt/lustre/work/hennig/hmx892/repos/bo-arena

# Set cache and tmp directories for Singularity
export SINGULARITY_CACHEDIR=$SCRATCH
export SINGULARITY_TMPDIR=$SCRATCH
export APPTAINER_CACHEDIR=$SCRATCH
export APPTAINER_TMPDIR=$SCRATCH

# Build the Singularity container
mkdir -p images
singularity build images/bencher.sif docker://gaunab/bencher:latest
