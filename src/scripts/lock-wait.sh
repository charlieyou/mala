#!/usr/bin/env bash
# Wait for and acquire a lock on a file.
# Usage: lock-wait.sh <filepath> [timeout_seconds] [poll_interval_ms]
# Requires: LOCK_DIR, AGENT_ID environment variables
# Optional: REPO_NAMESPACE for cross-repo disambiguation
# Exit: 0 if lock acquired, 1 if timeout, 2 on error

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MALA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: lock-wait.sh <filepath> [timeout_seconds] [poll_interval_ms]" >&2
    exit 2
fi

filepath="$1"
timeout="${2:-30}"
poll_ms="${3:-100}"

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

exec env PYTHONPATH="$MALA_ROOT" python -m src.infra.tools.locking wait "$filepath" "$timeout" "$poll_ms"
