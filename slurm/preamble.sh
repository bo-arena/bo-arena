# Shared preamble for SLURM batch scripts.
# Source this at the top of each .sbatch file:
#   source "$(pwd)/slurm/preamble.sh"

# SLURM copies batch scripts to /var/spool/slurm/... before running them,
# so we locate config.sh via pwd (set to the repo root by submit.sh --chdir).
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
