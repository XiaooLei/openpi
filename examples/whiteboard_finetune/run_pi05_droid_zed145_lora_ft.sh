#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUN_DIR="/inspire/qb-ilm/project/gjjproject/public/xl/wipeboard/pi05_zed145_finetune"
if [[ -d "$DEFAULT_RUN_DIR" ]]; then
  RUN_DIR="${OPENPI_WHITEBOARD_RUN_DIR:-$DEFAULT_RUN_DIR}"
else
  RUN_DIR="${OPENPI_WHITEBOARD_RUN_DIR:-$SCRIPT_DIR}"
fi
export OPENPI_CONFIG="pi05_droid_whiteboard_zed145_joint_position_lora_finetune"
"$SCRIPT_DIR/run_pi05_droid_zed145_ft.sh" "${1:-zed145_joint_position_lora_$(date +%Y%m%d-%H%M%S)}" "${@:2}"
