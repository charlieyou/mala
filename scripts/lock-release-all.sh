#!/usr/bin/env bash
# Release all locks held by current agent.
# Usage: lock-release-all.sh
# Requires: LOCK_DIR, AGENT_ID environment variables

set -euo pipefail

if [[ -z "${LOCK_DIR:-}" ]] || [[ -z "${AGENT_ID:-}" ]]; then
    echo "Error: LOCK_DIR and AGENT_ID must be set" >&2
    exit 2
fi

for lock in "$LOCK_DIR"/*.lock; do
    if [[ -f "$lock" ]] && [[ "$(cat "$lock" 2>/dev/null)" = "$AGENT_ID" ]]; then
        rm -f "$lock"
    fi
done

exit 0
