#!/usr/bin/env bash
#SBATCH --job-name=aissai_hackathon
#SBATCH --reservation=pdl18409_oct2026
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --time=01:00:00

set -x

# Fix Linux stack size to prevent NCCL communication segfaults
ulimit -s 8192

# Ensure log directory exists (Slurm fails if the folder does not exist)
mkdir -p slurm_logs

# Execute code inside Apptainer container
srun apptainer exec --nv \
  --bind  /lustre/work/pdl18409/shared:/workspace \
  /lustre/work/pdl18409/shared/containers/surrogate_models.sif \
  bash run.sh

