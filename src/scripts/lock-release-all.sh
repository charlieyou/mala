#!/usr/bin/env bash
# Release all locks held by current agent.
# Usage: lock-release-all.sh
# Requires: LOCK_DIR, AGENT_ID environment variables

set -euo pipefail

exec uv run python -m src.tools.locking release-all
