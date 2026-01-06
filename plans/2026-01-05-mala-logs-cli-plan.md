# Implementation Plan: mala logs CLI Command

## Context & Goals
- **Spec**: `docs/2026-01-05-mala-logs-cli-spec.md`
- Add `mala logs` CLI subcommand group for searching run metadata and session logs
- Three subcommands: `list`, `sessions --issue <id>`, `show <run-id>`
- Human-readable table by default, `--json` flag for JSON output
- Read-only operations with current-repo scoping by default
- Improves debugging and historical tracking without manual JSON inspection

## Scope & Non-Goals
- **In Scope**
  - `mala logs list`: Show 20 most recent runs with summary stats
  - `mala logs sessions --issue <id>`: Find sessions by issue ID
  - `mala logs show <run-id>`: Show detailed run metadata
  - Human-readable table output (default)
  - JSON output via `--json` flag
  - Current repo scoping (default) and `--all` flag for all repos
  - Run-id prefix matching (8-char minimum)
  - JSONL log path display for direct inspection
- **Out of Scope (Non-Goals)**
  - Log deletion/rotation/modification
  - Real-time streaming
  - Advanced filtering beyond `--issue`
  - Time-based filtering (future)

## Assumptions & Constraints

### Implementation Constraints
- **Module organization**: Separate module `src/cli/logs.py` with logs-specific logic, registered with main app
- Must follow existing Typer patterns from `cli.py`
- Must use lazy imports to avoid SDK loading until needed
- Must work without running mala instance (read-only)
- Use existing `encode_repo_path` and `get_runs_dir` utilities from infra layer
- Do not introduce heavy dependencies for simple JSON reading

### Testing Constraints
- Unit tests for pure logic (prefix matching, repo scoping)
- Integration tests with `tmp_path` for file-based scenarios
- No e2e tests needed (simple read-only commands)

## Prerequisites
- [x] Spec approved (`docs/2026-01-05-mala-logs-cli-spec.md`)
- [x] Existing data structures (`RunMetadata`, `IssueRun`) are sufficient
- [x] Access to `src/infra/tools/env.py` (`encode_repo_path`, `get_runs_dir`)
- [x] Access to `src/infra/io/log_output/run_metadata.py` (`RunMetadata`)
- [x] `typer` available in environment (existing dependency)
- [ ] Add `tabulate` to project dependencies in `pyproject.toml`

## High-Level Approach
**Single PR implementation** - the feature is self-contained and can be delivered atomically.

The implementation will add a new `logs` subcommand group to the mala CLI. The core logic involves:
1. Scanning run metadata JSON files in the runs directory
2. Filtering by repo path (using `encode_repo_path` matching)
3. Parsing and formatting output (table or JSON)
4. Handling edge cases (corrupt files, ambiguous prefixes)

## Technical Design

### Architecture
**Separate module pattern**: Create `src/cli/logs.py` with a Typer sub-app, register it in `cli.py` via `app.add_typer(logs_app, name="logs")`.

**Layers:**
- **CLI Layer**: `src/cli/logs.py` handles argument parsing, output formatting, and error handling
- **Data Access**: Direct file system access using `pathlib` and `json` to read metadata
- **Utils**: Reuse `src/infra/tools/env.py` for path resolution (`get_runs_dir()`, `get_repo_runs_dir()`, `encode_repo_path()`)

**Data Flow:**
1. CLI receives command + args
2. Resolve CWD via `Path.cwd().resolve()` and compute expected repo subdir name via `encode_repo_path(Path.cwd().resolve())` — **Note**: Uses literal CWD, same as `mala run` default behavior (not git root). Users should run from project root for consistent results.
3. For repo-scoped queries (no `--all`), use `get_repo_runs_dir(Path.cwd().resolve())` to get the target directory directly
4. For `--all`, discover metadata files via `get_runs_dir().rglob("*.json")`
5. **Performance optimization for `list`**: Sort files by filename (descending) since filenames are `{timestamp}_{run_id[:8]}.json`. Parse files in order and stop after collecting 20 valid runs.
6. Parse each file using lightweight `json.load()` + field extraction (not full `RunMetadata.load()` for `list`/`sessions`)
7. Validate JSON has required keys (`run_id`, `started_at`, `issues`) - skip silently if not valid run metadata
8. Apply command-specific logic (sorting by `started_at` for final ordering, filtering, aggregation)
9. For `list`: take first 20 results after sorting
10. Format output (table via `tabulate` or JSON via `json.dumps`)

### Data Model
Uses existing data structures:
- `RunMetadata` from `src/infra/io/log_output/run_metadata.py`
- `IssueRun` dataclass with `issue_id`, `session_id`, `status`, `log_path`
- `encode_repo_path()` from `src/infra/tools/env.py`

**Run metadata JSON schema** (read-only):
```json
{
  "run_id": "uuid-string",
  "started_at": "iso-datetime",
  "config": { ... },
  "issues": {
    "issue-id": {
      "issue_id": "string",
      "session_id": "uuid-string | null",
      "status": "success | failed | timeout",
      "log_path": "string | null"
    }
  }
}
```

**JSON validation**: Files must contain `run_id`, `started_at`, `issues` keys to be considered valid run metadata. Files lacking these keys (e.g., debug logs, future formats) are silently skipped (no warning).

### API/Interface Design

**Commands:**
```
mala logs list [--json] [--all]
mala logs sessions --issue <id> [--json] [--all]
mala logs show <run-id> [--json]
```

**Table Output Columns:**
- `list`: run_id (full UUID), started_at, issue_count, success, fail, timeout, metadata_path
- `sessions`: run_id, session_id, issue_id, run_started_at, status, log_path
- `show`: Key-value formatted output of all metadata fields
- With `--all`: adds `repo_path` column to output

**Table Formatting:**
- Use `tabulate` with `tablefmt="simple"` style (clean, minimal borders, matches existing mala CLI aesthetics)
- Timestamps displayed in local time (ISO format)
- Null values shown as `-`
- Sort order: `started_at` desc (parsed as datetime), then `run_id` asc for determinism

**Output Schemas (JSON):**
- `list`: `[{run_id, started_at, issue_count, success_count, fail_count, timeout_count, metadata_path, repo_path?}]`
- `sessions`: `[{run_id, session_id, issue_id, run_started_at, status, log_path, repo_path?}]`
- `show`: Full run metadata structure via `serialize_run_metadata(data)` helper (wraps `RunMetadata._to_dict()` or equivalent direct JSON)
- Errors: `{"error": "<code>", "message": "<details>", "matches": [...]?}` (matches array for ambiguous prefix)

**Run-id Matching:**
- Accept full UUID or 8-char prefix
- **Filename format**: `{ISO-timestamp}_{run_id[:8]}.json` (e.g., `2026-01-05T10-30-00_a1b2c3d4.json`)
- **Matching strategy**:
  1. Extract run-id prefix from filename: split on `_`, take last segment before `.json`
  2. If input is 8 chars, match against filename prefix directly
  3. If input is full UUID, match filename prefix first, then verify `run_id` field in JSON
  4. **Fallback**: If filename parsing fails, fall back to scanning JSON `run_id` field
- If prefix matches multiple runs: exit 1 with error listing all matches
- If no match found: exit 1

**Exit Codes:**
- 0: Success (including empty results)
- 1: Run-id not found or ambiguous prefix
- 2: Run-id matched but file unreadable/corrupt (applies to `show` command only)

**Error and Warning Output Rules:**
- **With `--json`**:
  - Errors emit JSON object to **stdout**: `{"error": "<code>", "message": "<details>", "matches": [...]?}`
  - Exit with appropriate non-zero code
  - Warnings (corrupt files during scan) still go to stderr (separate from JSON output)
- **Without `--json`** (table mode):
  - Errors print human-readable message to stderr
  - Warnings (corrupt files) also print to stderr
- **Exit code 2 scope**: Only applies to `show` command when the matched file cannot be read. For `list`/`sessions`, corrupt files are skipped with warning (exit 0 if any valid results)

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/cli/cli.py` | Exists | Add `app.add_typer(logs_app, name="logs")` registration |
| `src/cli/logs.py` | **New** | Logs subcommand implementation (~150-200 LOC) |
| `pyproject.toml` | Exists | Add `tabulate` to dependencies |
| `tests/unit/cli/test_logs.py` | **New** | Unit tests for prefix matching, scoping |
| `tests/integration/cli/test_logs_integration.py` | **New** | Integration tests with tmp_path fixtures |

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes

| Case | Applies To | Handling | Exit Code |
|------|------------|----------|-----------|
| Corrupt JSON (parse error) | `list`, `sessions` | Skip file, warn to stderr, continue | 0 (if any valid) |
| Corrupt JSON (parse error) | `show` | Error with details (matched file unreadable) | 2 |
| Missing required keys | `list`, `sessions` | Silently skip (not run metadata) | 0 |
| Missing required keys | `show` | Error (matched file is not valid run metadata) | 2 |
| Ambiguous run-id prefix | `show` | Error listing all matches | 1 |
| Run-id not found | `show` | Error "no matching run" | 1 |
| Empty results | All | "No runs found" message / `[]` JSON | 0 |
| Null `session_id`/`log_path` | All | Show `-` in table, `null` in JSON | 0 |
| No matching repo subdir | `list`, `sessions` | Empty results (not an error) | 0 |
| Missing runs directory | All | Treat as empty (no runs) | 0 |
| All files corrupt | `list`, `sessions` | Empty results + warnings to stderr | 0 |

### Risks
- **Performance**: Scanning hundreds of JSON files is acceptable for expected scale (~100s of runs). For `show`, use filename prefix to narrow candidates before full JSON parse. For `sessions`, scan depth limited to most recent 100 runs per repo if needed.
- **Memory**: Using lightweight JSON parsing (extract only needed fields) mitigates memory concerns. Only `show` needs full metadata via `RunMetadata._to_dict()`.
- **Concurrent writes**: If mala is actively writing a run metadata file, read may see partial JSON. Treat as corrupt (skip with warning).

### Breaking Changes
- None (new feature, no existing behavior modified)
- New dependency: `tabulate` (no conflicts expected)

## Testing & Validation Strategy

### Unit Tests (`tests/unit/cli/test_logs.py`)
- Run-id prefix matching logic (exact match, 8-char prefix, ambiguous detection, no match)
- `encode_repo_path` scoping (filter by CWD matches subdir name)
- Output formatting helpers (table row generation, JSON serialization)
- Null value handling in output
- Sorting logic (started_at desc, run_id asc tie-breaker)
- Issue ID exact matching (case-sensitive)

### Integration Tests (`tests/integration/cli/test_logs_integration.py`)
- `tmp_path` with synthetic `RunMetadata` JSON files:
  - Valid runs with various statuses
  - Runs with null session_id/log_path values
  - Corrupt JSON files (parse errors)
  - Non-run JSON files (missing required keys)
- `--json` output shape validation (assert JSON schema)
- `--all` behavior (includes all repos, adds repo_path column)
- Corrupt JSON handling (skip + warning to stderr)
- Empty results (exit 0, "No runs found" / valid JSON `[]`)
- Exit code verification for all error conditions

### Manual Verification
- Run `mala logs list` in dev environment to see actual past runs
- Verify table formatting and column widths
- Test `mala logs show <run-id>` with real run metadata
- Test `mala logs sessions --issue <id>` against known issues

### Acceptance Criteria Coverage
| Spec AC | Covered By |
|---------|------------|
| `list` shows 20 most recent runs | Unit: sorting logic; Integration: full flow |
| `sessions --issue` finds sessions | Unit: issue matching; Integration: multi-run scan |
| `show <run-id>` displays details | Integration: full metadata output |
| Run-id prefix matching | Unit: prefix/ambiguity tests |
| Default human-readable table | Integration: default output format |
| `--json` for JSON-only | Integration: JSON schema validation |
| `--all` for all repos | Integration: multi-repo fixtures |
| Exit codes (0/1/2) | Integration: exit code assertions |
| JSON error output | Integration: error format validation |
| Corrupt file handling | Integration: corrupt file fixture |
| Null value display | Unit + Integration: null handling |

## Implementation Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Module organization | Separate `src/cli/logs.py` | Keeps cli.py manageable, cleaner separation |
| Table formatting | `tabulate` with `tablefmt="simple"` | Clean minimal style matching existing mala CLI aesthetics |
| JSON parsing | Lightweight (`json.load()` + field extraction) | Faster, less memory for `list`/`sessions` commands |
| Full metadata load | `serialize_run_metadata()` helper for `show` | Wraps private `_to_dict()` to isolate from internal changes; alternatively read raw JSON and output directly |
| Run-id matching | Filename prefix first, then JSON field | Efficient narrowing without parsing all files |
| Ambiguity handling | Error with match list | Deterministic, fail-safe per spec |
| Repo scoping | CWD resolved path (same as `mala run`) | Consistent with existing behavior |
| Invalid JSON files | Silently skip if missing required keys | Graceful handling of non-run files |
| Corrupt JSON files | Warn to stderr, skip file | Best-effort scanning continues |
| Timestamp display | Local time (ISO format) | User-friendly while unambiguous |
| Sorting | `started_at` desc, then `run_id` asc | Deterministic ordering per spec |
| `sessions` scoping | Same as `list` (CWD default, `--all` available) | Consistent UX across commands |

## Open Questions
- **Performance of `sessions` search**: Searching all past runs for an issue ID might be slow with thousands of files. For MVP, scan all files in target repo directory. Consider adding `--limit N` or scan-depth optimization if performance becomes an issue.
- **Future `--limit N` flag**: Spec mentions as future consideration for `list` command.

## Next Steps
After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
