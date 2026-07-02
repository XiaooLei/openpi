#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUN_DIR="/inspire/qb-ilm/project/gjjproject/public/xl/wipeboard/pi05_zed145_finetune"
if [[ -d "$DEFAULT_RUN_DIR" ]]; then
  RUN_DIR="${OPENPI_WHITEBOARD_RUN_DIR:-$DEFAULT_RUN_DIR}"
else
  RUN_DIR="${OPENPI_WHITEBOARD_RUN_DIR:-$SCRIPT_DIR}"
fi
BASE_OPENPI_DIR="${BASE_OPENPI_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$SCRIPT_DIR/common_env.sh"
PYTHON_BIN="$(resolve_python_bin)"
CONFIG="${OPENPI_CONFIG:-pi05_droid_whiteboard_zed145_force_finetune}"
EXP_NAME="${1:-zed145_force_$(date +%Y%m%d-%H%M%S)}"
if [[ $# -gt 0 ]]; then
  shift
fi
PER_DEVICE_BATCH_SIZE="${OPENPI_PER_DEVICE_BATCH_SIZE:-32}"

export OPENPI_WHITEBOARD_196EPS_FORCE_DATASET="${OPENPI_WHITEBOARD_196EPS_FORCE_DATASET:-$RUN_DIR/data/wipe_board_196eps_force}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
DEFAULT_OPENPI_DATA_HOME="/inspire/qb-ilm/project/gjjproject/public/xl/openpi-baseline/.cache/openpi"
if [[ ! -d "$DEFAULT_OPENPI_DATA_HOME" ]]; then
  DEFAULT_OPENPI_DATA_HOME="$BASE_OPENPI_DIR/.cache/openpi"
fi
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$DEFAULT_OPENPI_DATA_HOME}"
export PYTHONPATH="$BASE_OPENPI_DIR/src:${PYTHONPATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

PI05_DROID_DIR="/inspire/qb-ilm/project/gjjproject/public/xl/openpi-baseline/.cache/openpi/openpi-assets/checkpoints/pi05_droid"
PI05_DROID_PARAMS="$PI05_DROID_DIR/params"
PI05_DROID_NORM_STATS="$PI05_DROID_DIR/assets/droid/norm_stats.json"
PALIGEMMA_TOKENIZER="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"

for path in \
  "$PYTHON_BIN" \
  "$PI05_DROID_PARAMS/_METADATA" \
  "$PI05_DROID_NORM_STATS" \
  "$PALIGEMMA_TOKENIZER" \
  "$OPENPI_WHITEBOARD_196EPS_FORCE_DATASET/meta/info.json"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 2
  fi
done

for arg in "$@"; do
  case "$arg" in
    --batch-size|--batch-size=*)
      cat >&2 <<EOF
This wrapper uses per-device batch size.

Set OPENPI_PER_DEVICE_BATCH_SIZE instead of passing OpenPI's global --batch-size.
Example:
  OPENPI_PER_DEVICE_BATCH_SIZE=32 $0 $EXP_NAME
EOF
      exit 2
      ;;
  esac
done

JAX_DEVICE_COUNT="$("$PYTHON_BIN" - <<'PY'
import jax

print(jax.device_count())
PY
)"
GLOBAL_BATCH_SIZE=$((PER_DEVICE_BATCH_SIZE * JAX_DEVICE_COUNT))
echo "config=$CONFIG"
echo "exp_name=$EXP_NAME"
echo "per_device_batch_size=$PER_DEVICE_BATCH_SIZE"
echo "jax_device_count=$JAX_DEVICE_COUNT"
echo "openpi_global_batch_size=$GLOBAL_BATCH_SIZE"
echo "dataset=$OPENPI_WHITEBOARD_196EPS_FORCE_DATASET"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints" "$RUN_DIR/assets"
cd "$BASE_OPENPI_DIR"

"$PYTHON_BIN" scripts/train.py "$CONFIG" \
  --exp-name "$EXP_NAME" \
  --checkpoint-base-dir "$RUN_DIR/checkpoints" \
  --assets-base-dir "$RUN_DIR/assets" \
  --batch-size "$GLOBAL_BATCH_SIZE" \
  --tensorboard-enabled \
  --no-wandb-enabled \
  --overwrite \
  "$@" \
  2>&1 | tee -a "$RUN_DIR/logs/${CONFIG}_${EXP_NAME}.log"
