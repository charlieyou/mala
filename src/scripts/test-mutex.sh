#!/usr/bin/env bash
# Global mutex wrapper for repo-wide commands (pytest, ruff, uv sync, etc.).
# Acquires a fixed mutex before running the command, releases on exit.
# Usage: test-mutex.sh <command> [args...]
# Requires: LOCK_DIR, AGENT_ID environment variables
# Exit: Command's exit code on success, non-zero with holder info if blocked

set -euo pipefail

# Fixed mutex key for all repo-wide commands
MUTEX_KEY="__test_mutex__"

if [[ -z "${LOCK_DIR:-}" ]]; then
    echo "Error: LOCK_DIR must be set" >&2
    exit 2
fi

if [[ -z "${AGENT_ID:-}" ]]; then
    echo "Error: AGENT_ID must be set" >&2
    exit 2
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: test-mutex.sh <command> [args...]" >&2
    exit 2
fi

# Get the directory containing this script for calling other lock scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to release the mutex
release_mutex() {
    "${SCRIPT_DIR}/lock-release.sh" "$MUTEX_KEY" 2>/dev/null || true
}

# Try to acquire the mutex
if ! "${SCRIPT_DIR}/lock-try.sh" "$MUTEX_KEY"; then
    holder=$("${SCRIPT_DIR}/lock-holder.sh" "$MUTEX_KEY" 2>/dev/null || echo "unknown")
    echo "Error: Mutex held by ${holder}" >&2
    exit 1
fi

# Set up trap to release mutex on exit (normal or signal)
trap release_mutex EXIT

# Run the command and capture its exit code
"$@"
