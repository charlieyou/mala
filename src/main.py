#!/usr/bin/env python3
"""
mala: Agent SDK orchestrator for parallel issue processing.

This module is a thin shim that exposes the CLI app from src.cli.
The actual implementation lives in src/cli.py.

Usage:
    mala run [OPTIONS] [REPO_PATH]
    mala clean
    mala status
"""

from .cli import app

if __name__ == "__main__":
    app()
