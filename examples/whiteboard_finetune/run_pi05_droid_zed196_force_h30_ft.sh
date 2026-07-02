#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OPENPI_CONFIG="${OPENPI_CONFIG:-pi05_droid_whiteboard_zed196_force_finetune_h30}"
exec "$SCRIPT_DIR/run_pi05_droid_zed196_force_ft.sh" "$@"
