#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  scripts/install_local.sh
fi

CONFIG="${IMG_CENSOR_CONFIG:-configs/local.yaml}"

exec .venv/bin/python scripts/download_models.py --config "$CONFIG" --stage prompt
