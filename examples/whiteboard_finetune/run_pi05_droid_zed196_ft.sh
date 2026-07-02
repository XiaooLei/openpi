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
CONFIG="${OPENPI_CONFIG:-pi05_droid_whiteboard_zed196_joint_position_finetune}"
REPO_ID="wipe_board_v1_zed196"
EXP_NAME="${1:-complete196_joint_position_$(date +%Y%m%d-%H%M%S)}"
if [[ $# -gt 0 ]]; then
  shift
fi
PER_DEVICE_BATCH_SIZE="${OPENPI_PER_DEVICE_BATCH_SIZE:-32}"
RESUME="${OPENPI_RESUME:-0}"
OVERWRITE_REQUESTED="0"

export OPENPI_WHITEBOARD_ZED196_DATASET="${OPENPI_WHITEBOARD_ZED196_DATASET:-$RUN_DIR/data/wipe_board_v1_zed196_force}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
DEFAULT_OPENPI_DATA_HOME="/inspire/qb-ilm/project/gjjproject/public/xl/openpi-baseline/.cache/openpi"
if [[ ! -d "$DEFAULT_OPENPI_DATA_HOME" ]]; then
  DEFAULT_OPENPI_DATA_HOME="$BASE_OPENPI_DIR/.cache/openpi"
fi
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$DEFAULT_OPENPI_DATA_HOME}"
export PYTHONPATH="$BASE_OPENPI_DIR/src:${PYTHONPATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

PI05_DROID_DIR="/inspire/qb-ilm/project/gjjproject/public/xl/openpi-baseline/.cache/openpi/openpi-assets/checkpoints/pi05_droid"
PI05_DROID_PARAMS="$PI05_DROID_DIR/params"
PALIGEMMA_TOKENIZER="/inspire/qb-ilm/project/gjjproject/public/xl/openpi-baseline/.cache/openpi/big_vision/paligemma_tokenizer.model"

if [[ "${SKIP_DATA_PREP:-0}" != "1" && ! -f "$OPENPI_WHITEBOARD_ZED196_DATASET/meta/info.json" ]]; then
  "$SCRIPT_DIR/prepare_zed196_dataset.sh"
fi

for path in \
  "$PYTHON_BIN" \
  "$PI05_DROID_PARAMS/_METADATA" \
  "$PALIGEMMA_TOKENIZER" \
  "$OPENPI_WHITEBOARD_ZED196_DATASET/meta/info.json"; do
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
    --resume)
      RESUME="1"
      ;;
    --overwrite)
      OVERWRITE_REQUESTED="1"
      ;;
  esac
done

if [[ "$RESUME" == "1" && "$OVERWRITE_REQUESTED" == "1" ]]; then
  echo "Cannot use --overwrite together with OPENPI_RESUME=1/--resume." >&2
  exit 2
fi

CHECKPOINT_MODE_ARGS=(--overwrite)
if [[ "$RESUME" == "1" ]]; then
  CHECKPOINT_MODE_ARGS=(--resume)
fi

JAX_DEVICE_COUNT="$("$PYTHON_BIN" - <<'PY'
import jax

print(jax.device_count())
PY
)"
GLOBAL_BATCH_SIZE=$((PER_DEVICE_BATCH_SIZE * JAX_DEVICE_COUNT))
echo "config=$CONFIG"
echo "exp_name=$EXP_NAME"
echo "per_device_batch_size=$PER_DEVICE_BATCH_SIZE"
echo "resume=$RESUME"
echo "jax_device_count=$JAX_DEVICE_COUNT"
echo "openpi_global_batch_size=$GLOBAL_BATCH_SIZE"
echo "openpi_data_home=$OPENPI_DATA_HOME"
echo "dataset=$OPENPI_WHITEBOARD_ZED196_DATASET"
echo "checkpoint_group=$RUN_DIR/checkpoints/$CONFIG"
echo "save_interval_steps=2000"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints" "$RUN_DIR/assets"

NORM_STATS="$RUN_DIR/assets/$CONFIG/$REPO_ID/norm_stats.json"
if [[ "${SKIP_NORM_STATS:-0}" != "1" && ! -f "$NORM_STATS" ]]; then
  echo "Computing norm stats: $NORM_STATS"
  cd "$RUN_DIR"
  "$PYTHON_BIN" "$BASE_OPENPI_DIR/scripts/compute_norm_stats.py" --config-name "$CONFIG"
fi

cd "$BASE_OPENPI_DIR"

"$PYTHON_BIN" scripts/train.py "$CONFIG" \
  --exp-name "$EXP_NAME" \
  --checkpoint-base-dir "$RUN_DIR/checkpoints" \
  --assets-base-dir "$RUN_DIR/assets" \
  --batch-size "$GLOBAL_BATCH_SIZE" \
  --tensorboard-enabled \
  --no-wandb-enabled \
  "${CHECKPOINT_MODE_ARGS[@]}" \
  "$@" \
  2>&1 | tee -a "$RUN_DIR/logs/${CONFIG}_${EXP_NAME}.log"
