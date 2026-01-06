"""Tests for lazy SDK import behavior.

These tests verify that `claude_agent_sdk` is NOT imported when:
1. `import src` is executed
2. `from src.orchestration.orchestrator import MalaOrchestrator` is executed
3. `from src.infra.hooks import ...` is executed

This ensures that bootstrap() runs before any SDK code loads.
"""

import subprocess
import sys
from pathlib import Path

# Compute repo root dynamically (tests/unit/ is two levels below repo root)
REPO_ROOT = Path(__file__).parent.parent.parent


def test_import_src_does_not_load_sdk() -> None:
    """Verify `import src` does NOT trigger claude_agent_sdk import."""
    # Run in a subprocess to get a clean import state
    code = """
import sys
# Clear any existing imports
for mod in list(sys.modules.keys()):
    if mod.startswith('claude_agent_sdk'):
        del sys.modules[mod]

import src
if any(mod.startswith('claude_agent_sdk') for mod in sys.modules):
    print('FAIL: claude_agent_sdk was imported')
    sys.exit(1)
print('PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "PASS" in result.stdout


def test_import_hooks_does_not_load_sdk() -> None:
    """Verify `from src.infra.hooks import ...` does NOT trigger claude_agent_sdk import."""
    code = """
import sys
# Clear any existing imports
for mod in list(sys.modules.keys()):
    if mod.startswith('claude_agent_sdk'):
        del sys.modules[mod]

from src.infra.hooks import block_dangerous_commands, DANGEROUS_PATTERNS
if any(mod.startswith('claude_agent_sdk') for mod in sys.modules):
    print('FAIL: claude_agent_sdk was imported')
    sys.exit(1)
print('PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "PASS" in result.stdout


def test_import_orchestrator_class_does_not_load_sdk() -> None:
    """Verify `from src.orchestration.orchestrator import MalaOrchestrator` does NOT trigger SDK import."""
    code = """
import sys
# Clear any existing imports
for mod in list(sys.modules.keys()):
    if mod.startswith('claude_agent_sdk'):
        del sys.modules[mod]

from src.orchestration.orchestrator import MalaOrchestrator
if any(mod.startswith('claude_agent_sdk') for mod in sys.modules):
    print('FAIL: claude_agent_sdk was imported')
    sys.exit(1)
print('PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "PASS" in result.stdout


def test_orchestrator_lazy_export_via_getattr() -> None:
    """Verify that src.__getattr__ lazily loads MalaOrchestrator on first access."""
    code = """
import sys
# Clear any existing imports
for mod in list(sys.modules.keys()):
    if mod.startswith('src'):
        del sys.modules[mod]

import src
# Just importing src shouldn't load orchestrator
if 'src.orchestration.orchestrator' in sys.modules:
    print('FAIL: src.orchestration.orchestrator was imported on `import src`')
    sys.exit(1)

# Accessing MalaOrchestrator should trigger lazy load
cls = src.MalaOrchestrator
if 'src.orchestration.orchestrator' not in sys.modules:
    print('FAIL: src.orchestration.orchestrator was NOT imported after accessing MalaOrchestrator')
    sys.exit(1)

# But still should not have loaded claude_agent_sdk
if any(mod.startswith('claude_agent_sdk') for mod in sys.modules):
    print('FAIL: claude_agent_sdk was imported')
    sys.exit(1)

print('PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "PASS" in result.stdout
