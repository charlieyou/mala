# Code Health Check: 2025-12-30

## Summary

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 0 |
| Medium | 4 |
| Low | 3 |

---

## Issues

### [Medium] Build artifacts committed to source control

**Files**: `dist/mala-0.1.0-py3-none-any.whl`, `dist/mala-0.1.0.tar.gz`
**Category**: Hygiene
**Type**: chore

The `dist/` directory containing build artifacts is committed to the repository. Build artifacts should be regenerated from source and don't belong in version control; they bloat the repo and can become stale.

**Fix**: Add `dist/` to `.gitignore` and remove the committed artifacts with `git rm -r dist/`.

---

### [Medium] Duplicated get_mcp_servers function across modules

**Files**: `src/orchestrator.py:166`, `src/pipeline/run_coordinator.py:77`, `src/pipeline/agent_session_runner.py:354`
**Category**: Structure
**Type**: task

The `get_mcp_servers` function exists in multiple files with similar but not identical implementations. This duplication creates maintenance burden - changes must be made in multiple places and can drift.

**Fix**: Consolidate to a single canonical implementation in `src/orchestrator.py` and import from there in other modules.

---

### [Medium] Broad exception handlers swallowing errors

**Files**: `src/braintrust_integration.py:76,126,161,181`, `src/hooks.py:292,587`, `src/validation/lint_cache.py:184,195`
**Category**: Correctness
**Type**: task

Multiple `except Exception:` blocks catch all exceptions without logging or re-raising. While some are intentional (telemetry shouldn't break main flow), others may hide real bugs.

**Fix**: Audit each broad exception handler and either: (1) narrow to specific exceptions, (2) add logging, or (3) document why swallowing is intentional.

---

### [Medium] Assert statements used in production code paths

**Files**: `src/pipeline/agent_session_runner.py:611,661,807`, `src/orchestrator.py:583`, `src/config.py:125`
**Category**: Correctness
**Type**: task

`assert` statements are used in production code paths, not just tests. Asserts can be disabled with `python -O`, potentially causing unexpected None/null errors.

**Fix**: Replace asserts in production code with explicit `if` checks and raise appropriate exceptions (ValueError, RuntimeError) with descriptive messages.

---

### [Low] Backwards-compatibility re-export module could be removed

**Files**: `src/validation/command_runner.py:1-22`
**Category**: Dead Code
**Type**: chore

This module exists solely for backwards compatibility, re-exporting from `src/tools/command_runner.py`. If no external code depends on the old import path, this adds unnecessary indirection.

**Fix**: Search for imports from `src.validation.command_runner` and if none exist outside the codebase, remove the re-export module.

---

### [Low] NullTelemetryProvider uses pass statements for protocol methods

**Files**: `src/telemetry.py:20,111,114,117,120,123,136,151,214`
**Category**: AI Smell
**Type**: chore

Multiple methods in `NullTelemetryProvider` consist of just `pass`, which is correct for a null implementation but could be documented.

**Fix**: Add docstrings or inline comments explaining these are intentional no-ops for the null implementation.

---

### [Low] Large test files may benefit from splitting

**Files**: `tests/test_orchestrator.py` (3969 lines), `tests/test_quality_gate.py` (3005 lines), `tests/test_validation.py` (2868 lines)
**Category**: Structure
**Type**: chore

Several test files exceed 2000 lines, making navigation and maintenance difficult. This is not a correctness issue but impacts developer experience.

**Fix**: Consider splitting into logical submodules if files continue to grow.

---

## Recommended Priority

1. Remove build artifacts from git (5 min)
2. Consolidate `get_mcp_servers` implementations (30 min)
3. Audit and document exception handlers (1-2 hours)
