#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUN_DIR="/inspire/qb-ilm/project/gjjproject/public/xl/wipeboard/pi05_zed145_finetune"
if [[ -d "$DEFAULT_RUN_DIR" ]]; then
  RUN_DIR="${OPENPI_WHITEBOARD_RUN_DIR:-$DEFAULT_RUN_DIR}"
else
  RUN_DIR="${OPENPI_WHITEBOARD_RUN_DIR:-$SCRIPT_DIR}"
fi
ZIP_PATH="${OPENPI_WHITEBOARD_ZED145_ZIP:-/inspire/qb-ilm/project/gjjproject/public/xl/data/droid_whiteboard/wipe_board_v1_zed145_lerobot.zip}"
DATASET_ROOT="${OPENPI_WHITEBOARD_ZED145_DATASET:-$RUN_DIR/data/wipe_board_v1_zed145}"
ZIP_PREFIX="droid_whiteboard/wipe_board_v1_zed145"
BASE_OPENPI_DIR="${BASE_OPENPI_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
source "$SCRIPT_DIR/common_env.sh"
PYTHON_BIN="$(resolve_python_bin)"

if [[ -f "$DATASET_ROOT/meta/info.json" ]]; then
  "$PYTHON_BIN" - "$DATASET_ROOT/meta/info.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    info = json.load(f)
episodes = info.get("total_episodes")
if episodes != 145:
    raise SystemExit(f"expected 145 episodes, found {episodes}")
print(f"dataset already prepared: {sys.argv[1]} ({episodes} episodes)")
PY
  exit 0
fi

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Missing dataset zip: $ZIP_PATH" >&2
  exit 2
fi

mkdir -p "$(dirname "$DATASET_ROOT")"
TMP_DIR="${DATASET_ROOT}.tmp.$$"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Extracting $ZIP_PATH"
echo "Target: $DATASET_ROOT"
unzip -q "$ZIP_PATH" "${ZIP_PREFIX}/*" -d "$TMP_DIR"

if [[ ! -f "$TMP_DIR/$ZIP_PREFIX/meta/info.json" ]]; then
  echo "Zip did not contain expected LeRobot metadata: $ZIP_PREFIX/meta/info.json" >&2
  exit 3
fi

mv "$TMP_DIR/$ZIP_PREFIX" "$DATASET_ROOT"
"$PYTHON_BIN" - "$DATASET_ROOT/meta/info.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    info = json.load(f)
episodes = info.get("total_episodes")
frames = info.get("total_frames")
if episodes != 145:
    raise SystemExit(f"expected 145 episodes, found {episodes}")
print(f"prepared dataset: {sys.argv[1]} ({episodes} episodes, {frames} frames)")
PY
