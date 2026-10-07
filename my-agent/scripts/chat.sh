#!/usr/bin/env bash
# Start Painthaker's text chat (src/chat.py) on Ubuntu/WSL, from any directory.
#
# Always uses the Linux environment outside the repo (default
# ~/.virtualenvs/painthaker, override with PAINTHAKER_VENV), never the project's
# .venv, which belongs to Windows. Nothing is installed or exported globally.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${PAINTHAKER_VENV:-$HOME/.virtualenvs/painthaker}"
SETUP_CMD="cd \"$PROJECT_DIR\" && UV_PROJECT_ENVIRONMENT=\"$ENV_DIR\" uv sync --locked --python 3.11"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    echo "Painthaker's Linux environment was not found at: $ENV_DIR" >&2
    echo "Create it once with:" >&2
    echo "  $SETUP_CMD" >&2
    exit 1
fi

cd "$PROJECT_DIR"
export UV_PROJECT_ENVIRONMENT="$ENV_DIR"

# Read-only check that the environment matches uv.lock; it never installs.
if command -v uv >/dev/null 2>&1 && ! uv sync --locked --check --quiet 2>/dev/null; then
    echo "The environment at $ENV_DIR doesn't match uv.lock. Update it with:" >&2
    echo "  $SETUP_CMD" >&2
    exit 1
fi

exec "$ENV_DIR/bin/python" src/chat.py "$@"
