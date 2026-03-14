#!/bin/bash
#SBATCH --job-name=run_bencher_singularity
#SBATCH --mem=16G
#SBATCH --partition=cpu-galvani
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH -o /mnt/lustre/work/hennig/hmx892/logs/%j_%x_%a.out
#SBATCH --chdir=/mnt/lustre/work/hennig/hmx892/repos/bo-arena

BENCHER_HOST_FILE=/mnt/lustre/work/hennig/hmx892/repos/bo-arena/slurm/bencher_host.txt

cleanup() {
    echo "OFFLINE" > $BENCHER_HOST_FILE
}
trap cleanup EXIT TERM

# Write IP address so experiment jobs can connect (hostname may not resolve inside containers)
hostname -I | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1 > $BENCHER_HOST_FILE

# Run the Singularity container
singularity run images/bencher.sif
