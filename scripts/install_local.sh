#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -e ".[api,dev]"

mkdir -p models/hf-cache samples outputs tmp

cat <<'EOF'
Local environment is ready.

Start the API:
  scripts/run_local_api.sh

Optional: pre-download enabled local models:
  .venv/bin/python scripts/download_models.py --config configs/local.yaml
EOF
