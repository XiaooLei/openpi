#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUN_DIR="/inspire/qb-ilm/project/gjjproject/public/xl/wipeboard/pi05_zed145_finetune"
if [[ -d "$DEFAULT_RUN_DIR" ]]; then
  RUN_DIR="${OPENPI_WHITEBOARD_RUN_DIR:-$DEFAULT_RUN_DIR}"
else
  RUN_DIR="${OPENPI_WHITEBOARD_RUN_DIR:-$SCRIPT_DIR}"
fi
TAR_PATH="${OPENPI_WHITEBOARD_ZED196_TAR:-/inspire/qb-ilm/project/gjjproject/public/xl/data/droid_whiteboard/wipe_board_v1_zed196_force_lerobot.tar}"
DATASET_ROOT="${OPENPI_WHITEBOARD_ZED196_DATASET:-$RUN_DIR/data/wipe_board_v1_zed196_force}"
TAR_PREFIX="droid_whiteboard/wipe_board_v1_zed196_force"
BASE_OPENPI_DIR="${BASE_OPENPI_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$BASE_OPENPI_DIR/.venv/bin/python3}"

repair_scalar_force_columns() {
  "$PYTHON_BIN" - "$DATASET_ROOT" <<'PY'
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

root = Path(sys.argv[1])
columns = (
    "force_torque_force_norm",
    "force_torque_torque_norm",
    "force_torque_connected",
)
rewritten = []
for path in sorted((root / "data" / "chunk-000").glob("episode_*.parquet")):
    table = pq.read_table(path)
    changed = False
    for name in columns:
        if name not in table.column_names:
            continue
        idx = table.column_names.index(name)
        column = table[name].combine_chunks()
        if pa.types.is_floating(column.type):
            values = pc.cast(column, pa.float32())
            fixed = pa.FixedSizeListArray.from_arrays(values, 1)
            table = table.set_column(idx, name, fixed)
            changed = True
    if changed:
        pq.write_table(table, path)
        rewritten.append(path.name)

print(f"repaired_scalar_force_columns={len(rewritten)}")
if rewritten:
    print(f"first_repaired={rewritten[:5]}")
PY
}

if [[ -f "$DATASET_ROOT/meta/info.json" ]]; then
  "$PYTHON_BIN" - "$DATASET_ROOT/meta/info.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    info = json.load(f)
episodes = info.get("total_episodes")
frames = info.get("total_frames")
features = info.get("features", {})
if episodes != 196:
    raise SystemExit(f"expected 196 episodes, found {episodes}")
if "force_torque_wrench" not in features:
    raise SystemExit("expected force_torque_wrench feature in zed196 dataset")
print(f"dataset already prepared: {sys.argv[1]} ({episodes} episodes, {frames} frames)")
PY
  repair_scalar_force_columns
  exit 0
fi

if [[ ! -f "$TAR_PATH" ]]; then
  echo "Missing dataset tar: $TAR_PATH" >&2
  exit 2
fi

mkdir -p "$(dirname "$DATASET_ROOT")"
TMP_DIR="${DATASET_ROOT}.tmp.$$"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Extracting $TAR_PATH"
echo "Target: $DATASET_ROOT"
tar -xf "$TAR_PATH" -C "$TMP_DIR" "$TAR_PREFIX"

if [[ ! -f "$TMP_DIR/$TAR_PREFIX/meta/info.json" ]]; then
  echo "Tar did not contain expected LeRobot metadata: $TAR_PREFIX/meta/info.json" >&2
  exit 3
fi

mv "$TMP_DIR/$TAR_PREFIX" "$DATASET_ROOT"
"$PYTHON_BIN" - "$DATASET_ROOT/meta/info.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    info = json.load(f)
episodes = info.get("total_episodes")
frames = info.get("total_frames")
features = info.get("features", {})
if episodes != 196:
    raise SystemExit(f"expected 196 episodes, found {episodes}")
if "force_torque_wrench" not in features:
    raise SystemExit("expected force_torque_wrench feature in zed196 dataset")
print(f"prepared dataset: {sys.argv[1]} ({episodes} episodes, {frames} frames)")
PY
repair_scalar_force_columns
