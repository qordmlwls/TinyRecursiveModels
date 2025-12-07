#!/bin/bash
#SBATCH --job-name=trm_maze_l40
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --ntasks-per-node=4
#SBATCH --account=aip-jjin5
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --cpus-per-task=6
#SBATCH --mem=450G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/euijin1/slurm/%x-%j.out
#SBATCH --mail-user=euijin1@ualberta.ca
#SBATCH --mail-type=END,FAIL

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
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_API_KEY=${WANDB_API_KEY:-8f05ac03be73a9566c05daa8a2dd7a0dc5720534}
export TRM_DATA_ROOT="${TRM_DATA_ROOT:-${SCRATCH}/trm/data}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}
export TORCH_CPP_LOG_LEVEL=INFO

mkdir -p "${SCRATCH}/slurm"
cd "${SLURM_SUBMIT_DIR}"

detect_gpus() {
  local raw="${1:-}"
  [[ -z "${raw}" ]] && { echo ""; return; }
  [[ "${raw}" == *":"* ]] && raw="${raw##*:}"
  echo "${raw}"
}

if [[ -z "${GPUS:-}" ]]; then
  if [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
    GPUS="$(detect_gpus "${SLURM_GPUS_PER_NODE}")"
  elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    GPUS="$(detect_gpus "${SLURM_GPUS_ON_NODE}")"
  else
    GPUS=4
  fi
fi
export GPUS
export NODES=${SLURM_JOB_NUM_NODES:-1}

if [[ -z "${MASTER_ADDR:-}" ]]; then
  if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
  else
    MASTER_ADDR="$(hostname)"
  fi
fi
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}

RUN_NAME="${RUN_NAME:-trm_maze_l40_${SLURM_JOB_ID}}"
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-768}

cmd=(
  torchrun
  --nnodes "${NODES}"
  --nproc-per-node "${GPUS}"
  --rdzv_backend=c10d
  --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}"
  pretrain.py
  arch=trm
  data_paths="[${TRM_DATA_ROOT}/maze-30x30-hard-1k]"
  evaluators="[]"
  global_batch_size="${GLOBAL_BATCH_SIZE}"
  arch.L_layers=2
  arch.H_cycles=3
  arch.L_cycles=4
  "+run_name=${RUN_NAME}"
  ema=True
)

echo "Launching TRM Maze-Hard training with ${GPUS} L40S GPUs."
echo "Command: ${cmd[*]}"

exec "${cmd[@]}"
