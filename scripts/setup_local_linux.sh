#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .cache/tools
if [[ ! -x .cache/tools/uv-x86_64-unknown-linux-gnu/uv ]]; then
  curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz -o .cache/tools/uv.tar.gz
  tar -xzf .cache/tools/uv.tar.gz -C .cache/tools
fi
export UV_CACHE_DIR="/tmp/kernel-study-uv-cache"
UV="$PWD/.cache/tools/uv-x86_64-unknown-linux-gnu/uv"
"$UV" venv --allow-existing --python /usr/bin/python3 .venv-linux
"$UV" pip install --link-mode copy --python .venv-linux/bin/python torch --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --link-mode copy --python .venv-linux/bin/python numpy matplotlib pytest
.venv-linux/bin/python scripts/inspect_linux.py
