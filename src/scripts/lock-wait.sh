#!/usr/bin/env bash
# Wait for and acquire a lock on a file.
# Usage: lock-wait.sh <filepath> [timeout_seconds] [poll_interval_ms]
# Requires: LOCK_DIR, AGENT_ID environment variables
# Optional: REPO_NAMESPACE for cross-repo disambiguation
# Exit: 0 if lock acquired, 1 if timeout, 2 on error

set -euo pipefail

if [[ -z "${LOCK_DIR:-}" ]] || [[ -z "${AGENT_ID:-}" ]]; then
    echo "Error: LOCK_DIR and AGENT_ID must be set" >&2
    exit 2
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: lock-wait.sh <filepath> [timeout_seconds] [poll_interval_ms]" >&2
    exit 2
fi

filepath="$1"
timeout_seconds="${2:-30}"
poll_interval_ms="${3:-100}"

# Normalize path: resolve symlinks and convert to absolute canonical form
# Use realpath if available, fall back to readlink -f (Linux) or manual resolution
is_literal_key() {
    [[ "$1" == "__test_mutex__" ]]
}

normalize_path() {
    local path="$1"
    if command -v realpath >/dev/null 2>&1; then
        realpath -m "$path" 2>/dev/null || echo "$path"
    elif command -v readlink >/dev/null 2>&1 && readlink -f "$path" >/dev/null 2>&1; then
        readlink -f "$path" 2>/dev/null || echo "$path"
    else
        # Fallback: resolve relative paths manually
        if [[ "$path" = /* ]]; then
            echo "$path"
        else
            echo "$(pwd)/$path"
        fi
    fi
}

if ! is_literal_key "$filepath"; then
    filepath="$(normalize_path "$filepath")"
fi

# Build canonical key: namespace:filepath if REPO_NAMESPACE is set
if [[ -n "${REPO_NAMESPACE:-}" ]]; then
    key="${REPO_NAMESPACE}:${filepath}"
else
    key="${filepath}"
fi

# Hash the key to avoid collisions (e.g., 'a/b' vs 'a_b')
key_hash=$(printf "%s" "$key" | sha256sum | cut -c1-16)
lock="${LOCK_DIR}/${key_hash}.lock"

mkdir -p "$LOCK_DIR"

# Calculate deadline
start_time=$(date +%s)
deadline=$((start_time + timeout_seconds))

# Poll interval in seconds (for sleep)
poll_interval_sec=$(awk "BEGIN {printf \"%.3f\", $poll_interval_ms / 1000}")

while true; do
    # Try to acquire the lock atomically
    tmp="$(mktemp "$LOCK_DIR/.locktmp.${AGENT_ID}.XXXXXX")"
    printf "%s\n" "$AGENT_ID" > "$tmp"

    if ln "$tmp" "$lock" 2>/dev/null; then
        rm -f "$tmp"
        exit 0
    fi
    rm -f "$tmp"

    # Check timeout
    current_time=$(date +%s)
    if [[ $current_time -ge $deadline ]]; then
        exit 1
    fi

    # Wait before retrying
    sleep "$poll_interval_sec"
done
