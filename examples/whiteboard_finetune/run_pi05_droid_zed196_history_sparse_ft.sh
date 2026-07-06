#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OPENPI_CONFIG="${OPENPI_CONFIG:-pi05_droid_whiteboard_zed196_joint_history_sparse_finetune}"
export OPENPI_REPO_ID="${OPENPI_REPO_ID:-wipe_board_v1_zed196_history_sparse}"
export OPENPI_EXP_PREFIX="${OPENPI_EXP_PREFIX:-complete196_joint_history_sparse}"

exec "$SCRIPT_DIR/run_pi05_droid_zed196_ft.sh" "$@"
