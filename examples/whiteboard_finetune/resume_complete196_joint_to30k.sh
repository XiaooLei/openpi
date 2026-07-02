#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export OPENPI_RESUME=1
export TRAIN_VARIANT=complete196
export OPENPI_PER_DEVICE_BATCH_SIZE="${OPENPI_PER_DEVICE_BATCH_SIZE:-32}"

TARGET_STEPS="${OPENPI_RESUME_TARGET_STEPS:-30000}"
LR_DECAY_STEPS="${OPENPI_RESUME_LR_DECAY_STEPS:-10000}"

exec ./run_job.sh complete196_joint_position \
  --num-train-steps "$TARGET_STEPS" \
  --lr-schedule.decay-steps "$LR_DECAY_STEPS"
