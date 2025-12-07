#!/bin/bash
#SBATCH --job-name=tr2_maze_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:8
#SBATCH --ntasks-per-node=8
#SBATCH --account=aip-jjin5
#SBATCH --cpus-per-task=8
#SBATCH --mem=1500G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/euijin1/slurm/%x-%j.out
#SBATCH --mail-user=euijin1@ualberta.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Avoid SKIP_CC_CVMFS undefined errors when sourcing the module environment
export SKIP_CC_CVMFS=${SKIP_CC_CVMFS:-0}
export FORCE_CC_CVMFS=${FORCE_CC_CVMFS:-0}
export ZSH_VERSION=${ZSH_VERSION:-0}
export RSNT_LD_LIBRARY_PATH=${RSNT_LD_LIBRARY_PATH:-}
export RSNT_SLURM_MPI_TYPE=${RSNT_SLURM_MPI_TYPE:-pmix}
export SLURM_MPI_TYPE=${SLURM_MPI_TYPE:-pmix}

# Ensure the module command is available
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh

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

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "ERROR: Virtual environment at ${VENV_PATH} is missing." >&2
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
    GPUS=8
  fi
fi
export GPUS
export NODES=${SLURM_JOB_NUM_NODES:-1}

if [[ -z "${MASTER_ADDR:-}" ]]; then
  if [[ -n "${SLURM_JOB_NODELIST:-}" ]] ; then
    MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
  else
    MASTER_ADDR="$(hostname)"
  fi
fi
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}

RUN_NAME="${RUN_NAME:-tr2_maze_h100_${SLURM_JOB_ID}}"
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}

CMD=(
  torchrun
  --nnodes "${NODES}"
  --nproc-per-node "${GPUS}"
  --rdzv_backend=c10d
  --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}"
  pretrain.py
  arch=tr2
  data_paths="[${TRM_DATA_ROOT}/maze-30x30-hard-1k]"
  evaluators="[]"
  epochs=50000
  eval_interval=5000
  min_eval_interval=5000
  lr=1e-4
  lr_min_ratio=0.1
  lr_warmup_steps=0
  weight_decay=1.0
  beta1=0.9
  beta2=0.999
  puzzle_emb_lr=1e-4
  puzzle_emb_weight_decay=1.0
  global_batch_size="${GLOBAL_BATCH_SIZE}"
  arch.L_layers=2
  arch.H_cycles=3
  arch.L_cycles=4
  arch.beam_size=2
  arch.branch_factor=2
  arch.policy_moves=16
  arch.search_prob=0.5
  "+run_name=${RUN_NAME}"
  ema=True
)

echo "Launching Maze-Hard TR2 training with ${GPUS} H100 GPUs."
echo "Command: ${CMD[*]}"

exec "${CMD[@]}"
