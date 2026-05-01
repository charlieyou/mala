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


def test_import_claude_provider_does_not_load_sdk() -> None:
    """Verify importing :class:`ClaudeAgentProvider` does NOT trigger
    ``claude_agent_sdk`` import.

    The provider bundles existing Claude pieces but their SDK imports are
    inside method bodies. A future refactor that hoists them to module top
    level would break the Amp-only run case (Amp users do not have the SDK
    installed) - this test rejects that.
    """
    code = """
import sys
for mod in list(sys.modules):
    if mod.startswith('claude_agent_sdk'):
        del sys.modules[mod]

from src.infra.clients.claude_provider import ClaudeAgentProvider
provider = ClaudeAgentProvider()
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


def test_import_claude_provider_does_not_load_amp_adapters() -> None:
    """Verify importing :class:`ClaudeAgentProvider` does NOT pull in any
    ``src.infra.clients.amp_*`` adapter module.

    This is the Amp/Claude isolation invariant: importing the Claude provider
    on a machine without Amp installed must succeed; in particular it must
    not eagerly load Amp adapter code that may import an Amp-only
    dependency in the future.
    """
    code = """
import sys
for mod in list(sys.modules):
    if mod.startswith('src.infra.clients.amp'):
        del sys.modules[mod]

from src.infra.clients.claude_provider import ClaudeAgentProvider
ClaudeAgentProvider()
amp_loaded = sorted(
    mod for mod in sys.modules
    if mod.startswith('src.infra.clients.amp')
)
if amp_loaded:
    print('FAIL: ' + ','.join(amp_loaded))
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


def test_import_amp_provider_does_not_load_sdk() -> None:
    """Verify importing :class:`AmpAgentProvider` does NOT trigger
    ``claude_agent_sdk`` import.

    The Amp provider must be importable on machines without the Claude
    SDK installed. A regression that hoists ``claude_agent_sdk`` to a
    top-level import inside ``amp_provider`` (or any of its compile-time
    deps) would break the Amp-only run case; this test rejects that.
    """
    code = """
import sys
for mod in list(sys.modules):
    if mod.startswith('claude_agent_sdk'):
        del sys.modules[mod]

from src.infra.clients.amp_provider import AmpAgentProvider
provider = AmpAgentProvider()
# Touching the lazy-init properties must NOT pull in the Claude SDK
# either; only an accidental top-level import would.
provider.client_factory  # noqa: B018
provider.log_provider  # noqa: B018
loaded = sorted(m for m in sys.modules if m.startswith('claude_agent_sdk'))
if loaded:
    print('FAIL: ' + ','.join(loaded))
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
