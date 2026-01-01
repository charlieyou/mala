#!/usr/bin/env python3
"""
mala: Agent SDK orchestrator for parallel issue processing.

This module is a thin shim that exposes the CLI app from src.cli.cli.
The actual implementation lives in src/cli/cli.py.

Usage:
    mala run [OPTIONS] [REPO_PATH]
    mala epic-verify [OPTIONS] EPIC_ID [REPO_PATH]
    mala clean
    mala status
"""

from .cli import bootstrap

# Call bootstrap at module import time so the console entrypoint (src.cli.main:app)
# runs bootstrap before importing the CLI app (which imports claude_agent_sdk)
bootstrap()

from .cli import app  # noqa: E402

if __name__ == "__main__":
    app()
