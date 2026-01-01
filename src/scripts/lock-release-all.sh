#!/usr/bin/env bash
# Release all locks held by current agent.
# Usage: lock-release-all.sh
# Requires: LOCK_DIR, AGENT_ID environment variables

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MALA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

exec env PYTHONPATH="$MALA_ROOT" python -m src.infra.tools.locking release-all
