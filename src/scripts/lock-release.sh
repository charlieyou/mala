#!/usr/bin/env bash
# Release a lock on a file (only if held by current agent).
# Usage: lock-release.sh <filepath>
# Requires: LOCK_DIR, AGENT_ID environment variables
# Optional: REPO_NAMESPACE for cross-repo disambiguation
# Exit: 0 if released or not held, 2 on error

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MALA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ $# -ne 1 ]]; then
    echo "Usage: lock-release.sh <filepath>" >&2
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

exec env PYTHONPATH="$MALA_ROOT" python -m src.infra.tools.locking release "$filepath"
