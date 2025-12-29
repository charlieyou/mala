#!/usr/bin/env bash
# Release all locks held by current agent.
# Usage: lock-release-all.sh
# Requires: LOCK_DIR, AGENT_ID environment variables

set -euo pipefail

exec uv run python -c "
import sys
sys.argv = ['locking', 'release-all']
from src.tools.locking import _cli_main
sys.exit(_cli_main())
"
