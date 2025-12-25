#!/usr/bin/env bash
# Try to acquire a lock on a file.
# Usage: lock-try.sh <filepath>
# Requires: LOCK_DIR, AGENT_ID environment variables
# Exit: 0 if lock acquired, 1 if already locked

set -euo pipefail

if [[ -z "${LOCK_DIR:-}" ]] || [[ -z "${AGENT_ID:-}" ]]; then
    echo "Error: LOCK_DIR and AGENT_ID must be set" >&2
    exit 2
fi

if [[ $# -ne 1 ]]; then
    echo "Usage: lock-try.sh <filepath>" >&2
    exit 2
fi

filepath="$1"
lock="${LOCK_DIR}/${filepath//\//_}.lock"

mkdir -p "$LOCK_DIR"

# Fast-path if already locked
if [[ -f "$lock" ]]; then
    exit 1
fi

# Atomic lock creation using hardlink to a temp file (no stale mutex dirs)
tmp="$(mktemp "$LOCK_DIR/.locktmp.${AGENT_ID}.XXXXXX")"
printf "%s\n" "$AGENT_ID" > "$tmp"

if ln "$tmp" "$lock" 2>/dev/null; then
    rm -f "$tmp"
    exit 0
fi

rm -f "$tmp"
exit 1
