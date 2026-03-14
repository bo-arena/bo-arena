#!/bin/bash
#SBATCH --job-name=run_bencher_singularity
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00

BENCHER_HOST_FILE=$(pwd)/slurm/bencher_host.txt

cleanup() {
    echo "OFFLINE" > $BENCHER_HOST_FILE
}
trap cleanup EXIT TERM

# Write IP address so experiment jobs can connect (hostname may not resolve inside containers)
hostname -I | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1 > $BENCHER_HOST_FILE

# Run the Singularity container
singularity run images/bencher.sif
