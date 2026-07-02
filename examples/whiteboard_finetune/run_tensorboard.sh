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
TENSORBOARD_BIN="$(resolve_tensorboard_bin)"
LOGDIR="${1:-$RUN_DIR/checkpoints}"
PORT="${TENSORBOARD_PORT:-6006}"

if [[ ! -x "$TENSORBOARD_BIN" ]]; then
  echo "Missing tensorboard binary: $TENSORBOARD_BIN" >&2
  exit 2
fi

mkdir -p "$LOGDIR"
echo "tensorboard_logdir=$LOGDIR"
echo "tensorboard_port=$PORT"
exec "$TENSORBOARD_BIN" --logdir "$LOGDIR" --host 0.0.0.0 --port "$PORT"
