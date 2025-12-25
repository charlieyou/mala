#!/usr/bin/env bash
# Get the agent ID holding a lock on a file.
# Usage: lock-holder.sh <filepath>
# Requires: LOCK_DIR environment variable
# Output: Agent ID holding the lock, or empty if unlocked

set -euo pipefail

if [[ -z "${LOCK_DIR:-}" ]]; then
    echo "Error: LOCK_DIR must be set" >&2
    exit 2
fi

if [[ $# -ne 1 ]]; then
    echo "Usage: lock-holder.sh <filepath>" >&2
    exit 2
fi

filepath="$1"
lock="${LOCK_DIR}/${filepath//\//_}.lock"

cat "$lock" 2>/dev/null || true
