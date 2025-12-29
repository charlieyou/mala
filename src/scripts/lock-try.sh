#!/usr/bin/env bash
# Try to acquire a lock on a file.
# Usage: lock-try.sh <filepath>
# Requires: LOCK_DIR, AGENT_ID environment variables
# Optional: REPO_NAMESPACE for cross-repo disambiguation
# Exit: 0 if lock acquired, 1 if already locked, 2 on error

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: lock-try.sh <filepath>" >&2
    exit 2
fi

# Normalize path to absolute (mimics realpath -m behavior)
filepath="$1"
if command -v realpath >/dev/null 2>&1; then
    filepath=$(realpath -m "$filepath" 2>/dev/null || echo "$filepath")
elif [[ "$filepath" != /* ]]; then
    filepath="$(pwd)/$filepath"
fi

exec uv run python -c "
import sys
sys.argv = ['locking', 'try', '$filepath']
from src.tools.locking import _cli_main
sys.exit(_cli_main())
"
