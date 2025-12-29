#!/usr/bin/env bash
# Check if current agent holds the lock on a file.
# Usage: lock-check.sh <filepath>
# Requires: LOCK_DIR, AGENT_ID environment variables
# Optional: REPO_NAMESPACE for cross-repo disambiguation
# Exit: 0 if locked by me, 1 otherwise, 2 on error

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: lock-check.sh <filepath>" >&2
    exit 2
fi

filepath="$1"

# Skip normalization for literal keys (non-path identifiers like __test_mutex__)
is_literal_key() {
    [[ "$1" == __*__ ]]
}

if ! is_literal_key "$filepath"; then
    # Normalize path to absolute (mimics realpath -m behavior)
    if command -v realpath >/dev/null 2>&1; then
        filepath=$(realpath -m "$filepath" 2>/dev/null || echo "$filepath")
    elif [[ "$filepath" != /* ]]; then
        filepath="$(pwd)/$filepath"
    fi
fi

if command -v uv >/dev/null 2>&1; then
    exec uv run python -m src.tools.locking check "$filepath"
else
    exec python -m src.tools.locking check "$filepath"
fi
