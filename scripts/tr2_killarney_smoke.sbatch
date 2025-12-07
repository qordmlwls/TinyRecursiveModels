#!/bin/bash
#SBATCH --job-name=tr2_smoke_l40
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=aip-jjin5
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=0-01:00:00
#SBATCH --partition=gpubase_l40s_b2
#SBATCH --output=/scratch/euijin1/slurm/%x-%j.out

set -euo pipefail

module --force purge
module load StdEnv/2023
module load cuda/12.6
module load cudnn
module load python/3.11.5

export PATH="/cm/shared/apps/slurm/current/bin:${PATH}"
export CUDA_HOME="${EBROOTCUDA}"

if [[ -z "${VENV_PATH:-}" ]]; then
  if [[ -d "${SLURM_SUBMIT_DIR}/.venv" ]]; then
    VENV_PATH="${SLURM_SUBMIT_DIR}/.venv"
  else
    echo "ERROR: Set VENV_PATH to your TinyRecursiveModels virtualenv." >&2
    exit 2
  fi
fi

source "${VENV_PATH}/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_API_KEY=${WANDB_API_KEY:-8f05ac03be73a9566c05daa8a2dd7a0dc5720534}
export TRM_DATA_ROOT="${TRM_DATA_ROOT:-${SCRATCH}/trm/data}"
export DISABLE_COMPILE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}
export TORCH_CPP_LOG_LEVEL=INFO

mkdir -p "${SCRATCH}/slurm"
cd "${SLURM_SUBMIT_DIR}"

export GPUS=1
export NODES=1
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
export RDZV_ENDPOINT="${MASTER_ADDR:-localhost}:${MASTER_PORT:-29500}"

PRESET="${1:-arc1}"
shift || true

./scripts/run_tr2_test.sh "${PRESET}" "$@"
