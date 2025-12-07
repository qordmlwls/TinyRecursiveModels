#!/bin/bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1

if [[ $# -lt 1 ]]; then
  cat <<'USAGE' >&2
Usage: ./scripts/run_tr2_test.sh <preset> [extra hydra overrides]
USAGE
  exit 1
fi

PRESET="$1"
shift || true

DATA_ROOT="${TRM_DATA_ROOT:-data}"
RAW_GPUS="${GPUS:-1}"
NODES="${NODES:-1}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:0}"
BEAM_SIZE="${BEAM_SIZE:-2}"
BRANCH_FACTOR="${BRANCH_FACTOR:-2}"
HALT_THRESHOLD="${HALT_THRESHOLD:-0.7}"
HALT_BONUS="${HALT_BONUS:-0.1}"
POLICY_WEIGHT="${POLICY_WEIGHT:-0.1}"
INTER_WEIGHT="${INTER_WEIGHT:-0.5}"
DIV_WEIGHT="${DIV_WEIGHT:-0.01}"
VALUE_WEIGHT="${VALUE_WEIGHT:-0.1}"
SEARCH_PROB="${SEARCH_PROB:-0.25}"
GLOBAL_BATCH_OVERRIDE="${GLOBAL_BATCH_SIZE:-128}"

if [[ "${RAW_GPUS}" == *:* ]]; then
  RAW_GPUS="${RAW_GPUS##*:}"
fi
if ! [[ "${RAW_GPUS}" =~ ^[0-9]+$ ]]; then
  echo "Failed to parse GPU count from '${RAW_GPUS}'." >&2
  exit 2
fi
GPUS="${RAW_GPUS}"

common_overrides=(
  "arch=tr2"
  "arch.beam_size=${BEAM_SIZE}"
  "arch.branch_factor=${BRANCH_FACTOR}"
  "arch.halt_threshold=${HALT_THRESHOLD}"
  "arch.halt_bonus=${HALT_BONUS}"
  "arch.policy_weight=${POLICY_WEIGHT}"
  "arch.inter_weight=${INTER_WEIGHT}"
  "arch.div_weight=${DIV_WEIGHT}"
  "arch.value_weight=${VALUE_WEIGHT}"
  "arch.search_prob=${SEARCH_PROB}"
)

case "${PRESET}" in
  arc1)
    RUN_NAME="${RUN_NAME:-pretrain_tr2_test_arc1_${GPUS}}"
    if (( GPUS > 1 )); then
      CMD=(torchrun --nnodes "${NODES}" --nproc-per-node "${GPUS}" --rdzv_backend=c10d --rdzv_endpoint "${RDZV_ENDPOINT}" pretrain.py)
    else
      CMD=(python pretrain.py)
    fi
    OVERRIDES=(
      "${common_overrides[@]}"
      "data_paths=[${DATA_ROOT}/arc1concept-aug-1000]"
      'arch.L_layers=2'
      'arch.H_cycles=3'
      'arch.L_cycles=4'
      "global_batch_size=${GLOBAL_BATCH_OVERRIDE}"
      "+run_name=${RUN_NAME}"
      'ema=True'
    )
    ;;
  *)
    echo "Unknown preset: ${PRESET}" >&2
    exit 1
    ;;
esac

if [[ $# -gt 0 ]]; then
  OVERRIDES+=("$@")
fi

echo "Launching test preset '${PRESET}' with ${GPUS} GPU(s) per node."
echo "Command: ${CMD[*]} ${OVERRIDES[*]}"

exec "${CMD[@]}" "${OVERRIDES[@]}"
