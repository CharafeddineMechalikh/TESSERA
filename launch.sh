#!/usr/bin/env bash
# Launch the Distributed Attention Lab on loopback only.
#   ./launch.sh          install dependencies, then start the server
#   ./launch.sh fast     start the server without installing dependencies
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PORT="${LAB_PORT:-8000}"

if [[ "${1:-}" != "fast" ]]; then
  echo "[launch] Installing pinned dependencies"
  python -m pip install -r requirements.txt
fi

echo "[launch] Generation runs through the distributed decoder, not Ollama"
echo "[launch] Serving on http://127.0.0.1:${PORT}"
exec python -m uvicorn app.main:app --host 127.0.0.1 --port "${PORT}"
