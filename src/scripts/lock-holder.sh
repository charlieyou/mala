#!/usr/bin/env bash
# Get the agent ID holding a lock on a file.
# Usage: lock-holder.sh <filepath>
# Requires: LOCK_DIR environment variable
# Optional: REPO_NAMESPACE for cross-repo disambiguation
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

# Normalize path: resolve symlinks and convert to absolute canonical form
# Use realpath if available, fall back to readlink -f (Linux) or manual resolution
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

filepath="$(normalize_path "$filepath")"

# Build canonical key: namespace:filepath if REPO_NAMESPACE is set
if [[ -n "${REPO_NAMESPACE:-}" ]]; then
    key="${REPO_NAMESPACE}:${filepath}"
else
    key="${filepath}"
fi

# Hash the key to avoid collisions (e.g., 'a/b' vs 'a_b')
key_hash=$(printf "%s" "$key" | sha256sum | cut -c1-16)
lock="${LOCK_DIR}/${key_hash}.lock"

cat "$lock" 2>/dev/null || true
