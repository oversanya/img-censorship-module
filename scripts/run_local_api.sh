#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  scripts/install_local.sh
fi

export IMG_CENSOR_CONFIG="${IMG_CENSOR_CONFIG:-configs/local.yaml}"
export IMG_CENSOR_MOCK="${IMG_CENSOR_MOCK:-0}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

exec .venv/bin/python -m uvicorn img_censor.api:app --host "$HOST" --port "$PORT" --log-level info
