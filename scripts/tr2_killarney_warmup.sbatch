#!/bin/bash
#SBATCH --job-name=tr2_killarney_arc1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:8
#SBATCH --ntasks-per-node=8
#SBATCH --account=aip-jjin5
#SBATCH --cpus-per-task=6
#SBATCH --mem=1500G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/euijin1/slurm/%x-%j.out
#SBATCH --mail-user=euijin1@ualberta.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Killarney exposes the standard Alliance module stack plus CUDA 12.x on the
# 8x H100 SXM nodes described at https://docs.alliancecan.ca/wiki/Killarney
module --force purge
module load StdEnv/2023
module load cuda/12.6
module load cudnn
module load python/3.11.5

# Ensure Slurm CLI utilities (scontrol/squeue) stay on PATH even after module purge
export PATH="/cm/shared/apps/slurm/current/bin:${PATH}"

export CUDA_HOME="${EBROOTCUDA}"

if [[ -z "${VENV_PATH:-}" ]]; then
  if [[ -d "${SLURM_SUBMIT_DIR}/.venv" ]]; then
    VENV_PATH="${SLURM_SUBMIT_DIR}/.venv"
  elif [[ -n "${PROJECT:-}" && -d "${PROJECT}/trm-env" ]]; then
    VENV_PATH="${PROJECT}/trm-env"
  else
    echo "ERROR: VENV_PATH not set and no default virtualenv found. Set VENV_PATH before submitting." >&2
    exit 2
  fi
fi

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "ERROR: Virtual environment at ${VENV_PATH} is missing. Check VENV_PATH." >&2
  exit 2
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
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}

mkdir -p "${SCRATCH}/slurm"

cd "${SLURM_SUBMIT_DIR}"

detect_gpus() {
  local raw="${1:-}"
  if [[ -z "${raw}" ]]; then
    echo ""
    return
  fi
  if [[ "${raw}" == *":"* ]]; then
    raw="${raw##*:}"
  fi
  echo "${raw}"
}

if [[ -z "${GPUS:-}" ]]; then
  if [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
    GPUS="$(detect_gpus "${SLURM_GPUS_PER_NODE}")"
  elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    GPUS="$(detect_gpus "${SLURM_GPUS_ON_NODE}")"
  elif [[ -n "${SLURM_GPUS:-}" ]]; then
    GPUS="$(detect_gpus "${SLURM_GPUS}")"
  else
    GPUS=8
  fi
fi
export GPUS

if [[ -z "${NODES:-}" ]]; then
  if [[ -n "${SLURM_NNODES:-}" ]]; then
    NODES="${SLURM_NNODES}"
  elif [[ -n "${SLURM_JOB_NUM_NODES:-}" ]]; then
    NODES="${SLURM_JOB_NUM_NODES}"
  elif [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    NODES="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | awk 'END{print NR}')"
  else
    NODES=1
  fi
fi
export NODES

export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-768}

if [[ -z "${MASTER_ADDR:-}" ]]; then
  if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
  else
    MASTER_ADDR="$(hostname)"
  fi
fi
export MASTER_ADDR

export MASTER_PORT=${MASTER_PORT:-29500}

if [[ -z "${RDZV_ENDPOINT:-}" ]]; then
  RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"
fi
export RDZV_ENDPOINT

if [[ $# -gt 0 ]]; then
  PRESET="$1"
  shift
else
  PRESET="${TR2_PRESET:-arc1}"
fi

EXTRA_OVERRIDES=("$@")
if [[ -n "${TR2_EXTRA_OVERRIDES:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_OVERRIDES+=(${TR2_EXTRA_OVERRIDES})
fi

if [[ -n "${RESUME_PATH:-}" ]]; then
  if [[ ! -f "${RESUME_PATH}" ]]; then
    echo "WARNING: RESUME_PATH '${RESUME_PATH}' not found; starting from scratch." >&2
  else
    EXTRA_OVERRIDES+=("+load_checkpoint=${RESUME_PATH}")
  fi
fi

./scripts/run_tr2_test.sh "${PRESET}" "${EXTRA_OVERRIDES[@]}"
