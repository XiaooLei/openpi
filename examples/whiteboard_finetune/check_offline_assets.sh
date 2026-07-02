#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUN_DIR="/inspire/qb-ilm/project/gjjproject/public/xl/wipeboard/pi05_zed145_finetune"
RUN_DIR="${OPENPI_WHITEBOARD_RUN_DIR:-$DEFAULT_RUN_DIR}"
BASE_OPENPI_DIR="${BASE_OPENPI_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

DEFAULT_OPENPI_DATA_HOME="/inspire/qb-ilm/project/gjjproject/public/xl/openpi-baseline/.cache/openpi"
if [[ ! -d "$DEFAULT_OPENPI_DATA_HOME" ]]; then
  DEFAULT_OPENPI_DATA_HOME="$BASE_OPENPI_DIR/.cache/openpi"
fi
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$DEFAULT_OPENPI_DATA_HOME}"

DATASET="${OPENPI_WHITEBOARD_ZED196_DATASET:-$RUN_DIR/data/wipe_board_v1_zed196_force}"
PI05_DROID_DIR="/inspire/qb-ilm/project/gjjproject/public/xl/openpi-baseline/.cache/openpi/openpi-assets/checkpoints/pi05_droid"

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "MISSING: $path" >&2
    return 1
  fi
  echo "OK: $path"
}

require_path "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
require_path "$PI05_DROID_DIR/params/_METADATA"
require_path "$PI05_DROID_DIR/params/manifest.ocdbt"
require_path "$PI05_DROID_DIR/params/d"
require_path "$PI05_DROID_DIR/assets/droid/norm_stats.json"

require_path "$DATASET/meta/info.json"
require_path "$DATASET/meta/episodes.jsonl"
require_path "$DATASET/meta/tasks.jsonl"
require_path "$DATASET/data/chunk-000"

EPISODES="$(wc -l < "$DATASET/meta/episodes.jsonl")"
if [[ "$EPISODES" != "196" ]]; then
  echo "MISSING: expected 196 episodes, found $EPISODES in $DATASET/meta/episodes.jsonl" >&2
  exit 1
fi
echo "OK: dataset episodes=$EPISODES"

require_path "$RUN_DIR/assets/pi05_droid_whiteboard_zed196_joint_position_finetune_h30/wipe_board_v1_zed196/norm_stats.json"
require_path "$RUN_DIR/assets/pi05_droid_whiteboard_zed196_force_finetune_h30/wipe_board_v1_zed196_force/norm_stats.json"

echo "OK: offline assets are ready"
