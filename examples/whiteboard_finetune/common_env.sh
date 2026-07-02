#!/usr/bin/env bash

resolve_python_bin() {
  if [[ -n "${OPENPI_PYTHON:-}" ]]; then
    echo "$OPENPI_PYTHON"
    return 0
  fi
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "$PYTHON_BIN"
    return 0
  fi

  local candidates=(
    "$BASE_OPENPI_DIR/.venv/bin/python3"
    "/inspire/qb-ilm/project/gjjproject/public/xl/projects/droid-whiteboard/openpi/.venv/bin/python3"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -x "$path" ]]; then
      echo "$path"
      return 0
    fi
  done

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  echo "Missing python. Set OPENPI_PYTHON=/path/to/python3." >&2
  return 2
}

resolve_tensorboard_bin() {
  if [[ -n "${OPENPI_TENSORBOARD:-}" ]]; then
    echo "$OPENPI_TENSORBOARD"
    return 0
  fi
  if [[ -n "${TENSORBOARD_BIN:-}" ]]; then
    echo "$TENSORBOARD_BIN"
    return 0
  fi

  local candidates=(
    "$BASE_OPENPI_DIR/.venv/bin/tensorboard"
    "/inspire/qb-ilm/project/gjjproject/public/xl/projects/droid-whiteboard/openpi/.venv/bin/tensorboard"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -x "$path" ]]; then
      echo "$path"
      return 0
    fi
  done

  if command -v tensorboard >/dev/null 2>&1; then
    command -v tensorboard
    return 0
  fi

  echo "Missing tensorboard. Set OPENPI_TENSORBOARD=/path/to/tensorboard." >&2
  return 2
}
