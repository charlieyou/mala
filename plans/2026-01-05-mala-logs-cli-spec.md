# Mala Logs CLI Command

**Tier:** S
**Owner:** [TBD]
**Target ship:** [TBD]
**Links:** Related: `src/cli/cli.py` (existing CLI), `src/infra/io/log_output/run_metadata.py` (run metadata)

## 1. Outcome & Scope

**Problem / context**
Currently there's no way to search mala logs, find sessions by issue ID, or retrieve metadata/stats from past runs without manually navigating JSON files in `~/.config/mala/runs/`. Developers debugging mala behavior or investigating past issue resolutions must manually parse JSONL logs and run metadata.

**Change summary**
Add a new `mala logs` CLI subcommand group with three subcommands: `list` (recent runs), `sessions --issue <id>` (find sessions by issue), and `show <run-id>` (run details). Default to human-readable table; add `--json` flag for JSON output.

**Scope boundary**
Read-only operations only. Explicitly out of scope: log deletion, log rotation, log modification, real-time streaming, advanced filtering beyond `--issue`.

**Storage layout and discovery**
- Run metadata stored at: `{get_runs_dir()}/{encode_repo_path(repo_path)}/{timestamp}_{run_id[:8]}.json`
- `get_runs_dir()`: `$MALA_RUNS_DIR` or `~/.config/mala/runs` (respects XDG_CONFIG_HOME)
- `encode_repo_path()`: prefix `-`, join path parts with `-`, normalize `_` to `-` (e.g., `/home/user/my_repo` → `-home-user-my-repo`)
- Current repo: use CWD resolved path (same as `mala run` default), NOT git root
- If CWD has no matching runs subdir: empty results (exit 0), not an error
- Discovery algorithm: scan `get_runs_dir()/**/*.json`, filter by matching `encode_repo_path(cwd)` to subdir name; `--all` skips filter

## 2. User Experience & Flows

**UX impact**
- User-visible? yes
- When user runs `mala logs <subcommand>`, they see human-readable table output by default
- Add `--json` flag for machine-parsable JSON output (pipe-friendly)
- Default scope: current repository (add `--all` flag for all repos)

## 3. Requirements + Verification

**Acceptance criteria**
- `mala logs list` shows the 20 most recent runs (by `RunMetadata.started_at`) with: run_id (full UUID), started_at, issue count, success/fail/timeout counts, metadata file path
- `mala logs sessions --issue <id>` finds all sessions for that issue ID (exact string match, case-sensitive), showing: run_id, session_id, issue_id, run_started_at, status, JSONL log path
- `mala logs show <run-id>` displays detailed metadata for a specific run including all sessions with their JSONL paths
- Run-id matching: accepts full UUID or 8-char prefix; if prefix is ambiguous, error listing matches (fail-safe)
- Default: human-readable table; `--json` flag for JSON-only output (pipe-friendly, no table)
- Default: current repo only; `--all` flag to search all repos (adds repo_path column to output)
- Output includes JSONL log file paths (from `IssueRun.log_path`) for direct log inspection
- Filtering: minimal (`--issue` only for sessions command)
- Empty state: exit 0 with "No runs found" message (valid JSON `[]` with `--json`)
- Corrupt files: best-effort scanning; skip with warning to stderr, continue, exit 0 if any valid results

**Exit codes**
- 0: Success (including empty results)
- 1: Run-id not found or ambiguous prefix
- 2: Run-id matched but file unreadable/corrupt

**Error output with `--json`**
- Errors always emit valid JSON to stdout: `{"error": "<code>", "message": "<details>", "matches": [...]?}`
- `matches` array included for ambiguous prefix errors (lists matching run_ids)
- Warnings (corrupt files skipped) still go to stderr

**Null value handling**
- `session_id` and `log_path` can be null in `IssueRun`
- Table output: show `-` for null values
- JSON output: use `null` literal
- Include rows with null values (don't filter them out)

## 4. Instrumentation & Release Checks

**Validation after release**
- How to confirm: Run `mala logs list` against a repo with prior mala runs; verify table and JSON appear; run `mala logs show <run-id>` and confirm paths are correct; verify `mala logs sessions --issue <id>` returns expected sessions
- Known risks: Read-only command, minimal risk

**Decisions made**
- Uses existing `RunMetadata` and `IssueRun` data structures from `src/infra/io/log_output/run_metadata.py`
- Follows existing CLI patterns in `src/cli/cli.py` (Typer framework)
- Output format: human-readable table by default; `--json` flag for JSON-only (pipe-friendly, no table mixed in)
- Repo scope: current repo only by default (CWD resolved, same as `mala run`); `--all` flag for all repos
- Run ordering: newest first (by `RunMetadata.started_at`, not filesystem mtime)
- Session timestamp: use `RunMetadata.started_at` as `run_started_at` (IssueRun has no per-session timestamp)
- Session status: from `IssueRun.status` field (success/failed/timeout)
- Run-id: accept full UUID or 8-char prefix; error on ambiguity with list of matches (deterministic, fail-safe)
- Default limit: 20 runs for `list` command
- Scanning strategy: parse all JSON files in target repo subdir (acceptable for expected scale ~100s of runs); `show` can use filename prefix to narrow candidates
- Corrupt file handling: best-effort (skip with stderr warning, continue scanning)
- Counts: issue_count = `len(issues)`, success/fail/timeout = count by `IssueRun.status` value
- session_id source: `IssueRun.session_id` field (Claude SDK session UUID, can be null)
- Issue matching: exact string match on `IssueRun.issue_id` (case-sensitive)
- JSON file identification: must parse and contain `run_id`, `started_at`, `issues` keys to be valid run metadata
- Ordering tie-breaker: `started_at` desc (parsed as datetime), then `run_id` asc for determinism

**JSON output schemas**
- `list --json`: `[{run_id, started_at, issue_count, success_count, fail_count, timeout_count, metadata_path, repo_path?}]`
- `sessions --json`: `[{run_id, session_id, issue_id, run_started_at, status, log_path, repo_path?}]`
- `show --json`: Full `RunMetadata._to_dict()` structure

**Test plan**
- Unit tests (`tests/unit/`): run-id prefix matching/ambiguity detection, `encode_repo_path` scoping logic
- Integration tests (`tests/integration/`): `tmp_path` with synthetic `RunMetadata` JSON files testing `--json` output shape, `--all` behavior, corrupt JSON handling, empty results, exit codes

**Open questions**
- Future: consider `--limit N` flag to override default 20
- Future: consider date filters, status filter, epic filter
