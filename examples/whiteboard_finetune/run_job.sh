#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUN_DIR="/inspire/qb-ilm/project/gjjproject/public/xl/wipeboard/pi05_zed145_finetune"
if [[ -d "$DEFAULT_RUN_DIR" ]]; then
  RUN_DIR="${OPENPI_WHITEBOARD_RUN_DIR:-$DEFAULT_RUN_DIR}"
else
  RUN_DIR="${OPENPI_WHITEBOARD_RUN_DIR:-$SCRIPT_DIR}"
fi
EXP_NAME="${1:-zed145_joint_position_$(date +%Y%m%d-%H%M%S)}"
if [[ $# -gt 0 ]]; then
  shift
fi

# User-facing knobs for job submission.
export TRAIN_VARIANT="${TRAIN_VARIANT:-full}"          # full | lora | complete196 | complete196_force | complete196_h50 | complete196_force_h50
export OPENPI_PER_DEVICE_BATCH_SIZE="${OPENPI_PER_DEVICE_BATCH_SIZE:-32}"
export RUN_TENSORBOARD="${RUN_TENSORBOARD:-1}"        # 1 | 0
export TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints"

TB_PID=""
cleanup() {
  if [[ -n "$TB_PID" ]] && kill -0 "$TB_PID" >/dev/null 2>&1; then
    kill "$TB_PID" >/dev/null 2>&1 || true
    wait "$TB_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "job_exp_name=$EXP_NAME"
echo "train_variant=$TRAIN_VARIANT"
echo "per_device_batch_size=$OPENPI_PER_DEVICE_BATCH_SIZE"
echo "run_tensorboard=$RUN_TENSORBOARD"
echo "tensorboard_port=$TENSORBOARD_PORT"

if [[ "$RUN_TENSORBOARD" == "1" ]]; then
  "$SCRIPT_DIR/run_tensorboard.sh" "$RUN_DIR/checkpoints" \
    >"$RUN_DIR/logs/tensorboard_${EXP_NAME}.log" 2>&1 &
  TB_PID="$!"
  echo "tensorboard_pid=$TB_PID"
  echo "tensorboard_log=$RUN_DIR/logs/tensorboard_${EXP_NAME}.log"
fi

case "$TRAIN_VARIANT" in
  full)
    TRAIN_SCRIPT="$SCRIPT_DIR/run_pi05_droid_zed145_ft.sh"
    ;;
  lora)
    TRAIN_SCRIPT="$SCRIPT_DIR/run_pi05_droid_zed145_lora_ft.sh"
    ;;
  complete196|zed196)
    TRAIN_SCRIPT="$SCRIPT_DIR/run_pi05_droid_zed196_ft.sh"
    ;;
  complete196_force|zed196_force|force)
    TRAIN_SCRIPT="$SCRIPT_DIR/run_pi05_droid_zed196_force_ft.sh"
    ;;
  complete196_h50|zed196_h50|h50)
    TRAIN_SCRIPT="$SCRIPT_DIR/run_pi05_droid_zed196_h50_ft.sh"
    ;;
  complete196_force_h50|zed196_force_h50|force_h50)
    TRAIN_SCRIPT="$SCRIPT_DIR/run_pi05_droid_zed196_force_h50_ft.sh"
    ;;
  *)
    echo "Unknown TRAIN_VARIANT=$TRAIN_VARIANT; expected full, lora, complete196, complete196_force, complete196_h50, or complete196_force_h50." >&2
    exit 2
    ;;
esac

set +e
"$TRAIN_SCRIPT" "$EXP_NAME" "$@"
STATUS="$?"
set -e

exit "$STATUS"
