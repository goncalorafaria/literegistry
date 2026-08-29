#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${XDG_RUNTIME_DIR:?}"
chmod 700 "$XDG_RUNTIME_DIR"

exec literegistry podman "$@"
