#!/usr/bin/env bash
# Release a lock on a file (only if held by current agent).
# Usage: lock-release.sh <filepath>
# Requires: LOCK_DIR, AGENT_ID environment variables
# Optional: REPO_NAMESPACE for cross-repo disambiguation
# Exit: 0 if released or not held, 1 on error

set -euo pipefail

if [[ -z "${LOCK_DIR:-}" ]] || [[ -z "${AGENT_ID:-}" ]]; then
    echo "Error: LOCK_DIR and AGENT_ID must be set" >&2
    exit 2
fi

if [[ $# -ne 1 ]]; then
    echo "Usage: lock-release.sh <filepath>" >&2
    exit 2
fi

filepath="$1"

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

if [[ -f "$lock" ]] && [[ "$(cat "$lock" 2>/dev/null)" = "$AGENT_ID" ]]; then
    rm -f "$lock"
fi

exit 0
