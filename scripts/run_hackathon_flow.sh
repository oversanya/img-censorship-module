#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/img-censor" ]]; then
  scripts/install_local.sh
fi

CONFIG="${IMG_CENSOR_CONFIG:-configs/local.yaml}"
PROMPT="${*:-Сгенерируй фото машины}"

exec .venv/bin/img-censor --config "$CONFIG" --stage full --prompt "$PROMPT"
