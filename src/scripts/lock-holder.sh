#!/usr/bin/env bash
# Get the agent ID holding a lock on a file.
# Usage: lock-holder.sh <filepath>
# Requires: LOCK_DIR environment variable
# Optional: REPO_NAMESPACE for cross-repo disambiguation
# Output: Agent ID holding the lock, or empty if unlocked

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MALA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ $# -ne 1 ]]; then
    echo "Usage: lock-holder.sh <filepath>" >&2
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

exec env PYTHONPATH="$MALA_ROOT" python -m src.infra.tools.locking holder "$filepath"
