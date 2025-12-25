#!/usr/bin/env bash
# Check if current agent holds the lock on a file.
# Usage: lock-check.sh <filepath>
# Requires: LOCK_DIR, AGENT_ID environment variables
# Exit: 0 if locked by me, 1 otherwise

set -euo pipefail

if [[ -z "${LOCK_DIR:-}" ]] || [[ -z "${AGENT_ID:-}" ]]; then
    echo "Error: LOCK_DIR and AGENT_ID must be set" >&2
    exit 2
fi

if [[ $# -ne 1 ]]; then
    echo "Usage: lock-check.sh <filepath>" >&2
    exit 2
fi

filepath="$1"
lock="${LOCK_DIR}/${filepath//\//_}.lock"

if [[ -f "$lock" ]] && [[ "$(cat "$lock" 2>/dev/null)" = "$AGENT_ID" ]]; then
    exit 0
fi

exit 1
