#!/usr/bin/env bash
# Release a lock on a file (only if held by current agent).
# Usage: lock-release.sh <filepath>
# Requires: LOCK_DIR, AGENT_ID environment variables
# Optional: REPO_NAMESPACE for cross-repo disambiguation
# Exit: 0 if released or not held, 2 on error

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: lock-release.sh <filepath>" >&2
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
sys.argv = ['locking', 'release', '$filepath']
from src.tools.locking import _cli_main
sys.exit(_cli_main())
"
