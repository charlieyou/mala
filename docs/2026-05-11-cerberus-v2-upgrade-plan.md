# Implementation Plan: Upgrade mala to Cerberus v2

## Context & Goals

- **Spec**: `/Users/cyou/code/cerberus/docs/2026-05-08-rebuild-spec.md` (cerberus side); no mala-side spec — derived from user description "upgrade cerberus to v2 - its a single go binary, assume that it will be on PATH"
- mala's reviewer/epic-verifier plumbing currently shells out to the cerberus v1 gate wrapper. Cerberus v2 ships as a single Go binary named `cerberus` with a related but **not byte-identical** CLI surface.
- Goal: replace the v1 integration with a v2 `cerberus` integration so that `reviewer_type: cerberus` routes through the v2 binary on `$PATH`, keeping `reviewer_type: agent_sdk` untouched.

## Scope & Non-Goals

- **In Scope**
  - Replace legacy gate CLI invocation with `cerberus` across mala's subprocess plumbing.
  - Update CLI flag/JSON parsing for v2 differences (binary name; `--max-rounds 0` is rejected by v2; v2 emits gate-state JSON with `status`/`verdict` fields, not v1's `consensus_verdict`/`aggregated_findings`).
  - Wire mala into v2's env contract (`CERBERUS_RUN_KEY`, `CERBERUS_HOST`, `CERBERUS_STATE_ROOT`, `CERBERUS_PROJECT_KEY`) so the v2 binary creates state correctly even though mala is `CERBERUS_HOST=generic`.
  - Extract findings from v2's on-disk artifacts (per-reviewer JSONs + iteration telemetry) since v2's `wait`/`status` JSON output no longer carries the aggregated findings array mala relied on in v1.
  - Remove plugin auto-discovery entirely; require `cerberus` on `$PATH`.
  - Use `src/infra/clients/cerberus_cli.py` and `src/infra/clients/cerberus_output_parser.py` after the client/parser renames. No re-export shims (per `CLAUDE.md` Code Migration Rules).
  - Update mala docs (`docs/project-config.md`, `docs/validation.md`, `docs/validation-triggers.md`, `docs/cli-reference.md`) to reference cerberus v2.
  - Update mala tests and fixtures that mock the `cerberus` binary.
- **Out of Scope (Non-Goals)**
  - Maintain dual v1/v2 support — clean break.
  - Exposing new v2 features through mala (rosters, `--debate`, `--reviewer`, `--consensus`) beyond pass-through via `spawn_args` / `wait_args`.
  - Changing mala's reviewer abstraction (`CodeReviewer` / `EpicVerificationModel` protocols) — those stay.
  - Changing the agent_sdk reviewer.
  - Migrating cerberus state between v1 and v2 — clean break per the v2 spec.

## Assumptions & Constraints

- v2 binary is named `cerberus` and is available on `$PATH` (user-stated assumption).
- The `claude` plugin install mechanism is no longer the canonical install path for cerberus; users provision the binary themselves.
- mala runs cerberus as `CERBERUS_HOST=generic` (no Stop-hook coordination); mala blocks on `cerberus wait` synchronously like it does today.
- v2 does NOT auto-resolve its install root from `argv[0]` / `os.Executable`. mala must set `CERBERUS_ROOT` explicitly (derived from `which("cerberus")` or user-provided) so the detached runtime can load `prompts/` and `config/`. **Layout assumption**: mala's auto-derivation expects `<CERBERUS_ROOT>/bin/cerberus` + `<CERBERUS_ROOT>/prompts/reviewers/`. This matches the cerberus plugin marketplace install (`~/.claude/plugins/cache/cerberus/cerberus/<version>/`) and `make install` from a checkout. It does **not** match bare `go install ./cmd/cerberus` — that drops only the binary into `$GOBIN` (typically `~/go/bin/cerberus`) without the support tree, so the derived root has no `prompts/reviewers/`. Bare-`go install` users, distro packages that split bin/share dirs, and any other non-standard layout MUST set `cerberus.env.CERBERUS_ROOT` (pointing at the checkout root) explicitly. `validate_binary()` surfaces a clear actionable error if the derived root is missing `prompts/reviewers/`. Documented in `docs/project-config.md`.

### Implementation Constraints

- Keep the `CodeReviewer` and `EpicVerificationModel` protocol surface unchanged — only the cerberus-side implementation changes.
- Keep the `reviewer_type: cerberus | agent_sdk` config knob; `cerberus` continues to mean "the bundled cerberus integration", which now points at v2.
- The mala.yaml `code_review.cerberus.{spawn_args, wait_args, env, timeout}` knobs remain — args/env now pass through to the v2 binary.

### Testing Constraints

- All existing cerberus-related tests under `tests/unit/infra/` and `tests/integration/orchestration/` MUST keep passing (rewritten where they assert v1 binary names/flags/JSON shapes).
- Add at least one integration test that exercises the v2 happy path against a fake `cerberus` shell-script binary on a per-test `PATH`.
- Maintain current 85% line coverage floor from `mala.yaml`; OQ-Plan-4 captures the case where the rewrite temporarily dips below.
- Prefer behavior assertions over call-count-only tests, per `tests/AGENTS.md`.

### Decision Log

| Decision | Rationale | Evidence | Tradeoff / Risk / Follow-up |
|----------|-----------|----------|------------------------------|
| Drop v1 entirely; `reviewer_type: cerberus` now means v2 | Clean break; the user is upgrading, not coexisting. Matches v2 spec's own "no migration" stance. | User decision; TODO.md line 1; cerberus v2 spec non-goals | Users on v1 must upgrade their cerberus install in lockstep with this mala release. Document in CHANGELOG. |
| Remove plugin auto-discovery; require `cerberus` on `$PATH` | User-stated assumption; cerberus v2 binary is host-neutral, not Claude-plugin-bound. | User decision; v2 README "single Go binary". | Anyone relying on auto-discovery of `.claude/plugins/cache/cerberus/...` must install on PATH instead. Document in CHANGELOG. |
| Extract findings by reading per-reviewer JSON from the v2 iteration directory | Mala-only change; no upstream cerberus PR required; matches v2 spec R8 telemetry layout. | User decision; v2 spec R8 (`<state_root>/<project>/<run>/iterations/<N>/`). | mala couples to cerberus's on-disk layout. Risk: cerberus reshuffles the path. Mitigation: locate findings via the `run_key` from gate-state.json plus a small path helper that can be updated centrally. |
| Set `CERBERUS_HOST=generic`, `CERBERUS_RUN_KEY`, `CERBERUS_STATE_ROOT`, `CERBERUS_PROJECT_KEY` explicitly from mala | mala needs to know where v2 wrote artifacts to extract findings; explicit env makes that deterministic and gives mala a clean dir to clean up. | User decision; v2 env contract. | mala owns one more pinned convention. Risk: cerberus changes the env contract — keep the env-builder in one place. |
| Drop `--max-rounds 0` from the spawn invocation; rely on the v2 single-pass default | v2 rejects `--max-rounds 0`; for non-`--debate` runs v2 is single-pass by default. mala drives its own iterative fix loop, so the daemon's auto-respawn never mattered. | `internal/cli/spawn_code_review.go:487`; v2 spec section 2 (Default review path). | If v2 changes the default to multi-round, mala will silently spawn extra rounds. Mitigation: integration test asserts a single reviewer invocation per `spawn`. |
| Rename the legacy CLI/parser modules to `cerberus_cli.py` and `cerberus_output_parser.py` | Both files are getting substantial rewrites in this PR; doing renames in the same change avoids future "v1 ghost name" confusion. | User decision. | Touches all import sites. Mitigation: ruff/grep sweep after rename; single PR. |
| Land everything in a single PR | Parser, env wiring, binary name, and tests are tightly coupled; splitting would leave intermediate states broken. | User decision. | Large PR diff. Mitigation: organized commits within the PR (rename commit → parser commit → CLI swap commit → tests → docs). |
| `CERBERUS_RUN_KEY = f"mala-{claude_session_id}"` (for code review) or `f"mala-{epic_session_token}"` (for epic verify); `CERBERUS_STATE_ROOT` under `MALA_RUNS_DIR` per mala run | Co-locates cerberus iteration outputs with mala state so a single mala-run cleanup wipes both. Uses the per-attempt session id (already passed/generated today) so the `CodeReviewer`/`EpicVerificationModel` protocols don't need new parameters. | User decision; v2 env contract; existing protocol surfaces (`CodeReviewer.__call__(claude_session_id=...)`, `CerberusEpicVerifier._generate_session_id()`). | Per-attempt uniqueness depends on session-id uniqueness (UUIDs / `token_hex(6)`); collisions are negligible. Empty-`claude_session_id` edge case is already rejected by `DefaultReviewer.__call__`. |
| No deprecation shim for `cerberus_bin_path`; programmatic `MalaConfig(cerberus_bin_path=…)` raises `TypeError` | Per `CLAUDE.md` "No backward-compatibility shims" rule. The field is not part of mala.yaml's schema (`config_parser.py` never accepted it) and `MALA_CERBERUS_BIN_PATH` is not a defined env var (auto-discovery populated the field, not user config). The only caller surface is the kwarg, and a `TypeError` is a fast, clear failure for that audience. | `src/domain/validation/config_parser.py` (no `cerberus_bin_path` in any `_*_FIELDS` set); `src/infra/io/config.py` (`MALA_CERBERUS_BIN_PATH` not read in `from_env`); `CLAUDE.md` Code Migration Rules. | A user who passes the kwarg gets `TypeError: unexpected keyword argument 'cerberus_bin_path'` rather than a deprecation warning. Mitigation: explicit CHANGELOG entry under "Breaking Changes". |

## Integration Analysis

### Existing Mechanisms Considered

| Existing Mechanism | Could Serve Feature? | Decision | Rationale |
|--------------------|---------------------|----------|-----------|
| `src/infra/clients/cerberus_cli.py` (subprocess wrapper) | Yes | **Extend** | Already encapsulates spawn/wait/resolve subprocess management. Swap binary name, update flags, keep dataclass shape. |
| `src/infra/clients/cerberus_review.py` (`DefaultReviewer`) | Yes | **Extend** | Orchestration logic (stale-gate recovery, retry, no-changes short-circuit) survives. Only the underlying CLI shape changes. |
| `src/infra/clients/cerberus_epic_verifier.py` (`CerberusEpicVerifier`) | Yes | **Extend** | Same pattern as DefaultReviewer; spawn-epic-verify + wait flow is preserved in v2. |
| `src/infra/clients/cerberus_output_parser.py` | Partial | **Rename + rewrite** | v2 gate-state JSON differs materially; `ReviewIssue`/`ReviewResult` dataclasses stay (consumers depend on them); parsing logic rewritten. Exit-code mapping simplified against v2 codes. |
| `src/infra/tools/cerberus.py` (plugin discovery helper) | No | **Delete** | Frozen decision: `cerberus` lives on `$PATH`; no fallback. Callers move to `shutil.which("cerberus", ...)`. |
| `src/core/protocols/infra.py` (env-config plugin discovery method) | Partial | **Replace/remove** | E2E preflight uses `shutil.which` instead; remove or rename the protocol method. Update contract tests and `tests/fakes/env_config.py`. |
| Protocol-based dispatch (`reviewer_type` config + factory) | Yes | **Reuse unchanged** | The plug point is correct; only the implementation behind `reviewer_type: cerberus` changes. |
| mala.yaml `cerberus.{spawn_args,wait_args,env,timeout}` config knobs | Yes | **Reuse unchanged** | Same shape; values pass through to the v2 binary. |
| `MalaConfig.cerberus_bin_path` field | No | **Delete** | Auto-discovery is gone; `$PATH` is the only resolution. Drop the field rather than leave a vestigial `None`-only attribute. |
| `src/pipeline/review_formatter.py` | Yes | **Reuse unchanged** | v2 findings still flow through `format_review_issues` to follow-up prompts. |
| New: per-reviewer iteration-findings reader | n/a | **New** | Small helper (`cerberus_iteration_findings.py`) that, given `(state_root, project_key, run_key)`, lists per-reviewer JSON files in the latest iteration and produces `ReviewIssue` records. Centralizing it lets future cerberus layout changes touch one file. |

### Integration Approach

- All cerberus-specific change is contained inside `src/infra/clients/cerberus_cli.py` (renamed), `src/infra/clients/cerberus_output_parser.py` (renamed), `src/infra/clients/cerberus_review.py`, `src/infra/clients/cerberus_epic_verifier.py`, and one new helper `src/infra/clients/cerberus_iteration_findings.py`. `src/infra/tools/cerberus.py` is deleted.
- No new abstraction layer; the existing protocol-based reviewer/verifier dispatch already isolates the integration.
- Binary location is `shutil.which("cerberus", path=effective_path)`. `MalaConfig.cerberus_bin_path` is removed.
- **State-root threading** (run-id available at construction time): the factory builds `DefaultReviewer` / `CerberusEpicVerifier` AFTER `RunMetadata.run_id` is assigned for the current mala run — both adapters are constructed inside `validation_triggers` setup, downstream of run-id allocation, so the factory can compute `state_root = Path(MALA_RUNS_DIR) / run_id / "cerberus"` and pass it as a constructor kwarg (same channel `bin_path` used today). `project_key` is also computed at factory time from `os.path.abspath(repo_path)` (or honored from `cerberus.env.CERBERUS_PROJECT_KEY` if the user pinned it). Adapters store both as instance attributes and pass them into every `CerberusCLI.build_env(...)` call. The reviewer/verifier protocols stay unchanged — no per-call run-id parameter needed. If a future refactor ever has to construct reviewers before run-id exists, the fallback is to pass a `Callable[[], Path]` (lazy resolver) instead of a fixed `Path`; this is **not** required by the current code path and is recorded only as a future-proofing note.
- The `code_review.cerberus.{spawn_args,wait_args,env}` config block stays; users with rosters/debate needs pass them via `spawn_args`.
- The env-builder in `CerberusCLI.build_env` is the single source for the v2 env contract; every spawn/wait/resolve subprocess call sources its env from this builder.
- Findings flow: cerberus writes per-reviewer JSONs into `<CERBERUS_STATE_ROOT>/<CERBERUS_PROJECT_KEY>/<CERBERUS_RUN_KEY>/iterations/<N>/`; the iteration-findings reader is called after `wait` returns and stitches issues into `ReviewResult`.
- mala continues to use the spawn-then-blocking-wait pattern; no Stop-hook integration (mala runs as `CERBERUS_HOST=generic`).

## Prerequisites

- [ ] Confirm v2 binary CLI surface frozen enough to target (`spawn-code-review`, `spawn-epic-verify`, `wait --json`, `status --json`, `resolve` per v2.0.0 README).
- [ ] Local `cerberus` binary available on `$PATH` for development and manual smoke tests (a fake shell-script binary is sufficient for CI).
- [x] Per-reviewer JSON layout pinned (OQ-Plan-1 resolved): `iterations/<N>/round-<R>/reviewers/<provider>#<index>/output.json`, `<N>`/`<R>` both 1-based, final round wins for debate.
- [ ] Resolve `CERBERUS_PROJECT_KEY` derivation strategy (OQ-Plan-2): defaulting to a SHA-256 prefix of the absolute repo root path unless overridden by user-supplied env.

## High-Level Approach

Single PR, organized as ordered commits. Each commit leaves the tree buildable; tests adjust within the same commit that changes their target.

1. **Rename commit** — rename the legacy CLI/parser modules to `cerberus_cli.py` and `cerberus_output_parser.py`; update all imports in source and tests; no logic change. Verifies the renames are clean (no shims, no stragglers).
2. **Iteration-findings reader commit** — add `src/infra/clients/cerberus_iteration_findings.py` (**New**) with a pure function that, given `(state_root, project_key, run_key)`, returns `(list[ReviewIssue], list[str parse_errors])` from the latest iteration directory; on-disk fixture unit tests cover happy path, empty dir, missing dir, malformed file skip, latest-iteration auto-selection.
3. **CLI + env-builder commit** — rewrite `cerberus_cli.py`: binary name `cerberus`, drop `--max-rounds 0`, expand `build_env` to set `CERBERUS_HOST=generic` + run/project/state keys; `_review_gate_bin()` → `_cerberus_bin()`; helper-name and docstring sweep; update `tests/unit/infra/test_cerberus_cli.py`.
4. **Parser commit** — rewrite `cerberus_output_parser.py` for v2 gate-state schema; `map_exit_code_to_result` reads `status`/`verdict` and combines with iteration-dir findings; map `requires_decision` per Decision Log; update `test_cerberus_output_parser.py`.
5. **Reviewer + epic verifier commit** — `DefaultReviewer` and `CerberusEpicVerifier` stop consulting `MalaConfig.cerberus_bin_path`, source findings from the iteration reader, build env via `CerberusCLI.build_env` with run_key/project_key/state_root derived from the active issue, and **delete the stale-gate recovery branch** (v2 doesn't surface "already active"; per-attempt RUN_KEY avoids collision). Update `test_cerberus_review.py` and `test_cerberus_epic_verifier.py`; remove the existing "already-active" recovery tests.
6. **Plugin-discovery removal commit** — delete `src/infra/tools/cerberus.py` and its tests; remove `MalaConfig.cerberus_bin_path`; remove or replace the env-config plugin discovery method on the protocol and update fakes/contract tests; update `factory.py` (`_check_review_availability`, `_check_epic_verifier_availability`) to use `shutil.which("cerberus", ...)` + a `prompts/reviewers/` existence check on the resolved root. No subcommand probe (v2 `spawn-epic-verify --help` can launch a real run; see Technical Design).
7. **Integration smoke commit** — add `tests/integration/clients/test_cerberus_v2_smoke.py` with a fake `cerberus` shell-script binary fixture covering happy path (verdict=pass with zero findings; verdict=fail with multi-reviewer findings; stale-gate retry; verdict=fail with empty iteration dir → parse error).
8. **Docs + TODO commit** — update `docs/project-config.md`, `docs/validation.md`, `docs/validation-triggers.md`, `docs/cli-reference.md`; remove TODO entry; add CHANGELOG entry calling out v2 prerequisite, `cerberus_bin_path` removal, plugin auto-discovery removal, and the `--max-rounds 0` rejection.

## Technical Design

### Architecture

mala's reviewer/verifier flow remains: orchestrator → factory selects `DefaultReviewer` or `AgentSDKReviewer` by `reviewer_type` → `DefaultReviewer` calls `CerberusCLI` (spawn + wait + resolve) → parses output → returns `ReviewResult`. v2 only changes the *bottom* of this stack (binary name, flags, JSON shapes, env contract); the rest is untouched.

```
Orchestrator
  └─► factory._create_code_reviewer (reviewer_type → DefaultReviewer or AgentSDKReviewer)
        └─► DefaultReviewer.__call__
              ├─► CerberusCLI.validate_binary()         # shutil.which("cerberus", …)
              ├─► CerberusCLI.build_env(...)            # CERBERUS_HOST=generic, RUN_KEY, STATE_ROOT, PROJECT_KEY, …
              ├─► CerberusCLI.spawn_code_review(...)    # `cerberus spawn-code-review --exclude .beads/ --commit <sha> …`
              ├─► CerberusCLI.wait_for_review(...)      # `cerberus wait --json --session-key $CERBERUS_RUN_KEY --timeout …`
              ├─► cerberus_output_parser.map_exit_code_to_result(...)
              │      ├─ parse v2 gate-state JSON → status, verdict, run_key, session_id, current_iteration
              │      └─ cerberus_iteration_findings.read_findings(state_root, project_key, run_key)
              │            └─ pick max `iterations/<N>/` (1-based), then max `round-<R>/`,
              │               then glob `reviewers/*/output.json` and parse each
              └─► ReviewResult(passed, issues, parse_error, fatal_error, review_log_path)
                    └─► format_review_issues(issues, base_path)  # follow-up prompt to coder
```

`CerberusEpicVerifier` follows the same pattern, mapping per-reviewer findings to `UnmetCriterion`.

### Data Model

No mala-side schema change. The two external schemas that change shape:

- **Cerberus v1 `wait --json` output** (today's mala parser input):
  - `status`: `pending | resolved | timeout | error | no_reviewers`
  - `consensus_verdict`: `PASS | FAIL | ERROR`
  - `aggregated_findings`: array of `{title, body, priority, file, line_start, line_end, reviewer, severity, ...}`
  - `parse_errors`: array
  - `session_dir`: filesystem path
- **Cerberus v2 `wait --json` output** (effectively `status --json`, since `wait` polls then calls `runStatus`):
  - `schema_version`, `run_key`, `host`, `project_key`, `session_id`, `transcript_path`
  - `status`: `pending | resolved`
  - `verdict`: `pass | fail | requires_decision` (lowercase) or `null`
  - `resolution_reason`, `current_iteration`, `max_rounds`, `debate`, `roster_id`, `started_at`, `ended_at`
  - **No `aggregated_findings` field.** Per-reviewer findings are persisted to `<state_root>/<project>/<run>/iterations/<N>/` per the v2 spec (R8).
- **Cerberus v2 per-reviewer JSON** (one file per reviewer per round per iteration):
  - Persisted at `<state_root>/<project>/<run>/iterations/<N>/round-<R>/reviewers/<instance_id>/output.json` where `<N>` is the 1-based iteration index, `<R>` is the 1-based round index, and `<instance_id>` is `<provider>#<index>` (e.g. `claude#1`, `gemini#1`, `codex#1`). Verified against `cerberus/internal/state/paths.go` (`IterationDir`, `RoundDir`, `ReviewerDir`, `WriteReviewerOutput`) and a real run under `~/.claude/projects/.../iterations/1/round-1/reviewers/claude#1/output.json`.
  - Sibling files in the reviewer directory: `prompt.md`, `stdout.log`, `stderr.log`, `telemetry.json` (not consumed by mala).
  - Single-pass reviews produce exactly one round (`round-1`). Debate runs produce `round-1..round-maxRounds`; reviewer outputs in the **highest-numbered `round-N` directory** are the canonical final outputs and the only ones mala should surface as findings. Earlier rounds are intermediate debate state.
  - There is no per-round `aggregate.json`; aggregation lives in-memory and lands in the top-level `gate-state.json`.
  - File payload: `findings: [{title, body, priority, file_path, line_start, line_end, confidence, severity}, ...]`, plus `verdict: str`, `summary: str`, `overall_confidence: float`, `strategy: str`, `round: int`, `peer_responses_seen: int`.

Internal types (`ReviewIssue`, `ReviewResult`, `UnmetCriterion`, `EpicVerdict`) keep their existing shape — only how they get populated changes. v2 `file_path` → `ReviewIssue.file`; `line_start`/`line_end`/`priority`/`title`/`body` map 1:1.

### API/Interface Design

No mala-public API change. Internal interfaces that change:

- `CerberusGateCLI` → `CerberusCLI` (file rename + class rename to drop the "gate" terminology):
  - `_cerberus_bin()` returns `"cerberus"` (PATH-only, no `bin_path` arg).
  - `validate_binary()` uses `shutil.which("cerberus", path=effective_path)` AND verifies the resolved `CERBERUS_ROOT` (per `build_env` resolution order) contains `prompts/reviewers/`. Returns `"cerberus binary not found in PATH"` or `"cerberus root <path> missing prompts/reviewers/"` accordingly.
  - `build_env(*, run_key, state_root, project_key=None, claude_session_id=None)` produces a merged dict including:
    - `CERBERUS_HOST=generic`
    - `CERBERUS_RUN_KEY=<run_key>` — supplied by the caller. Sourced as:
      - `DefaultReviewer.__call__` passes `run_key = f"mala-{claude_session_id}"`. `claude_session_id` is already a required argument on the `CodeReviewer` protocol (today's `cerberus_review.py` enforces it via the "CLAUDE_SESSION_ID missing" check), and mala generates a fresh session id per review attempt, so this is unique per `(issue, attempt)` without protocol changes.
      - `CerberusEpicVerifier.verify` passes `run_key = f"mala-{self._generate_session_id()}"` (the existing `epic-<token_hex>` generator already produces a per-attempt unique value).
      - **No `issue_id`/`attempt` parameters are added to `__call__`/`verify`.** The `CodeReviewer` / `EpicVerificationModel` protocol surfaces stay exactly as today.
    - `CERBERUS_STATE_ROOT=<state_root>` (under `MALA_RUNS_DIR/<run_id>/cerberus` per mala run) — supplied by the caller via the adapter's constructor (already how `bin_path` was wired today, so no protocol change).
    - `CERBERUS_PROJECT_KEY=<project_key>` — see OQ-Plan-2 for derivation. **Overridable by user** via `cerberus.env.CERBERUS_PROJECT_KEY` (see Project-key override note below); the env-builder honors the override and the iteration-findings reader consumes the final resolved value, so reads and writes agree.
    - `CERBERUS_ROOT=<resolved_root>` — required by v2 to locate `prompts/reviewers/*.md`, `prompts/strategies/*.md`, and `config/gemini-readonly-policy.toml`. v2's `internal/config.Resolve` and `internal/prompts.rootDir` read `CERBERUS_ROOT` (then `CLAUDE_PLUGIN_ROOT`, then `PLUGIN_ROOT`) and fall back to `os.Getwd()` — without it mala would spawn cerberus from the mala repo cwd and prompt loads would fail. Resolution order:
      1. Honor user override `cerberus.env.CERBERUS_ROOT` if set.
      2. Else honor `$CERBERUS_ROOT` from the parent process env.
      3. Else derive from `shutil.which("cerberus", path=effective_path)`: `Path(which_result).resolve().parent.parent` (v2 ships as `<root>/bin/cerberus`). Verify `<root>/prompts/reviewers/` exists; if not, fail with a clear actionable error before spawning.
    - **Mala-owned keys** that user `cerberus.env` cannot override: `CERBERUS_RUN_KEY` (mala must know where the gate lives to `wait` on it) and `CERBERUS_STATE_ROOT` (mala owns the run dir for cleanup). `CERBERUS_PROJECT_KEY`, `CERBERUS_ROOT`, and `CERBERUS_HOST` are overridable: user-supplied `cerberus.env` values for those keys win over mala defaults.
    - `CLAUDE_SESSION_ID` kept only when the caller passes one (v2 uses `CERBERUS_SESSION_ID` / explicit `--session-id`).
  - `spawn_code_review(...)` removes `--max-rounds 0` from the command list. Command: `cerberus spawn-code-review --exclude .beads/ [--context-file FILE] [spawn_args…] --commit SHA…`.
  - `spawn_epic_verify(...)` removes `--max-rounds 0`. Command: `cerberus spawn-epic-verify [spawn_args…] EPIC.md`.
  - `wait_for_review(...)` uses `--session-key <run_key> --timeout <secs>` where `<run_key>` is the same `CERBERUS_RUN_KEY` value built by `build_env`. v2 keys state directories by RunKey via `state.RunDir(state_root, project_key, run_key)`; `--session-key` sets `env.RunKey` directly in cerberus's `envWithRunOverrides`, whereas `--session-id` would force `env.RunKey = <session_id>` and miss the directory created under `CERBERUS_RUN_KEY`. Keeps `--finalize` as v2 accepts it as a reserved compatibility flag. mala stops threading a Claude session id into wait/status entirely — RunKey is the single addressing key.
  - `resolve_gate(...)` unchanged in shape; only the binary name changes.
  - `WaitResult` holds **only** `returncode`, raw `stdout`, raw `stderr`, and `timed_out` — the v1-era pre-parsed `session_dir` is removed. All JSON parsing happens in `cerberus_output_parser`; mala addresses cerberus state by `run_key` (not `session_id`), and the iteration directory is computed as `state_root / project_key / run_key / "iterations"` without needing any field on `WaitResult`. (The parser still extracts `session_id` from the gate-state JSON for telemetry; it is never used for addressing.) This avoids double-parsing the gate-state JSON in two places.
- `cerberus_output_parser.py` (renamed from `cerberus_output_parser.py`):
  - `ReviewIssue`, `ReviewResult` dataclasses unchanged.
  - `parse_gate_state(stdout: str) -> GateState` — new pure parser for v2 gate-state JSON.
  - `map_exit_code_to_result(...)` rewritten: on zero exit, parse the gate-state JSON, then call the iteration-findings reader for `(issues, parse_errors)`. **Fail-closed rules (applied in order):**
    1. If the iteration-findings reader returned any parse errors due to a **missing layer** (no `iterations/`, no `round-*`, no `reviewers/`, no `output.json`), force `passed=False` and surface the parse_error regardless of the gate-state `verdict`. A legitimate pass writes empty `findings:[]` inside present reviewer dirs, never missing dirs.
    2. If any **individual reviewer JSON** in the round failed to parse (malformed file inside an otherwise-populated `reviewers/` dir), surface that parse error via `ReviewResult.parse_error` even when other reviewers produced usable findings — a missing peer in a consensus model must invalidate the pass. With `verdict=pass`, the parse error forces `passed=False`.
    3. Otherwise derive `passed = (verdict == "pass")`. Map `requires_decision` to `passed=False, fatal_error=False` and synthesize one `ReviewIssue(title="Cerberus reviewers reached no consensus", body="Gate verdict=requires_decision. Inspect per-reviewer outputs under <iteration_dir>; human decision or re-run required.", priority=1)` so the follow-up prompt has actionable text rather than an empty issues list (prevents agent fix-loop on empty findings). The body uses only values mala already has — `iteration_dir` is the same path the findings reader walked — and avoids referencing `consensus_count`/`reviewer_count` which are not in the v2 gate-state schema.
- `cerberus_iteration_findings.py` (**New**):
  - `read_findings(state_root: Path, project_key: str, run_key: str, *, iteration: int | None = None, round_index: int | None = None) -> tuple[list[ReviewIssue], list[str]]` returning `(issues, parse_errors)`.
  - Traversal (pseudocode):
    1. `iter_dir = iterations/<iteration>` if `iteration` is set, else the largest integer-named child of `iterations/` (must be `>= 1`; `0` is not a valid v2 iteration).
    2. `round_dir = iter_dir/round-<round_index>` if set, else the largest integer-suffixed `round-<R>` child. This selects `round-1` for single-pass reviews and the final round for debate runs.
    3. Glob `round_dir/reviewers/*/output.json` (one file per reviewer instance; the reviewer directory name `<provider>#<index>` becomes the `reviewer` attribution on each `ReviewIssue`).
  - Missing `iterations/`, empty iterations dir, missing `round-*` dir, or a `reviewers/` directory containing zero `output.json` files all return `([], [<descriptive parse error>])` — never crash; the caller decides whether to fail-closed based on verdict.
  - Malformed individual `output.json` files are skipped with one parse error each; other reviewers in the same round are still returned.
  - `latest_iteration_dir(...)` and `latest_round_dir(...)` helpers exported for tests.
- `MalaConfig`:
  - Remove `cerberus_bin_path` field.
  - Remove the plugin discovery call in `from_env`.
  - Keep `cerberus_spawn_args`, `cerberus_wait_args`, `cerberus_env`, `review_timeout`.
- `factory._check_review_availability` and `_check_epic_verifier_availability`:
  - Build the resolved env via the same `cerberus.env`-aware logic the runtime uses (read `cerberus.env.CERBERUS_ROOT`, then process env, then PATH derivation), so a user-supplied `CERBERUS_ROOT` is honored at availability time and disable-reasons match what would actually fail at spawn time.
  - With that effective env, run `shutil.which("cerberus", path=effective_path)` and verify the resolved `CERBERUS_ROOT` has `prompts/reviewers/` (delegates to `CerberusCLI.validate_binary()` so the two paths can't drift).
  - **No subcommand probe.** Do NOT execute `cerberus spawn-epic-verify --help` (or `cerberus spawn-code-review --help`): v2 normalizes unknown dash-prefixed args on `spawn-epic-verify` as raw epic-criterion text, so `--help` can launch a real verification run during availability checking. Binary-presence + root-completeness is sufficient — if a subcommand is missing, the first real spawn surfaces the error.

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/infra/clients/cerberus_cli.py` (renamed from `cerberus_cli.py`) | Renamed + rewritten | Binary name `cerberus`; drop `--max-rounds 0`; rewrite `build_env` for v2 env contract; `WaitResult` reduced to `(returncode, stdout, stderr, timed_out)` (no pre-parsed gate-state fields — keeps JSON parsing in one place: the parser module); rename class to `CerberusCLI`; remove stale-gate spawn-retry orchestration helper. |
| `src/infra/clients/cerberus_output_parser.py` (renamed from `cerberus_output_parser.py`) | Renamed + rewritten | v2 gate-state schema; calls iteration-findings reader to populate `issues`; maps `requires_decision` per Decision Log. |
| `src/infra/clients/cerberus_iteration_findings.py` | **New** | Pure helper that walks the v2 nested layout `<state_root>/<project>/<run>/iterations/<N>/round-<R>/reviewers/<instance_id>/output.json` — selects the highest 1-based `<N>`, then the highest `<R>` (final debate round / sole single-pass round), then parses every `output.json`; returns `(issues, parse_errors)`. |
| `src/infra/clients/cerberus_review.py` | Exists | Import from renamed modules; remove `bin_path` usage; build env via `CerberusCLI.build_env(run_key=f"mala-{claude_session_id}", state_root=self.state_root, project_key=self.project_key, claude_session_id=claude_session_id)`; `state_root` and `project_key` are passed into `DefaultReviewer` at construction by the factory (same wiring channel `bin_path` used today); keep `CLAUDE_SESSION_ID`-required check (now also the basis for RUN_KEY); **remove stale-gate recovery code path** (v2 `StartSinglePass` warns-and-proceeds rather than failing with "already active" — see `cerberus/internal/orchestrator/orchestrator.go:90` and per-attempt-RUN_KEY collision-free design); keep no-changes short-circuit. |
| `src/infra/clients/cerberus_epic_verifier.py` | Exists | Same import + env changes; build env via `CerberusCLI.build_env(run_key=f"mala-{session_id}", state_root=self.state_root, project_key=self.project_key, claude_session_id=session_id)` where `session_id` is the existing `epic-<token>` value from `_generate_session_id()`; rewrite `_parse_wait_output` for v2 gate-state + iteration-findings; map per-reviewer findings to `UnmetCriterion`. |
| `src/infra/clients/agent_sdk_review.py` | Exists | **Update import** — line 25 currently does `from src.infra.clients.cerberus_output_parser import ReviewIssue, ReviewResult`; after the rename this becomes `from src.infra.clients.cerberus_output_parser import ReviewIssue, ReviewResult`. No-shim policy means this import must move in the same PR. Otherwise unchanged (no protocol metadata addition). |
| `src/infra/clients/__init__.py` | Exists | Update imports to renamed modules; no shims. |
| `src/infra/tools/cerberus.py` | **Deleted** | Plugin auto-discovery removed; `cerberus` lives on `$PATH`. |
| `src/infra/tools/env.py` | Exists | Remove/replace plugin discovery with PATH-based executable check if still needed by E2E. |
| `src/core/protocols/infra.py` | Exists | Remove or rename the env-config plugin discovery method; update fakes/contract tests. |
| `src/infra/io/config.py` | Exists | **Remove** `cerberus_bin_path` field, its construction in `from_env`, and the plugin discovery import. **Add** two new `MalaConfig` fields: `cerberus_state_root: Path | None = None` (factory-resolved default `MALA_RUNS_DIR/<run_id>/cerberus`) and `cerberus_project_key: str | None = None` (factory-resolved default `sha256(os.path.abspath(repo_path))[:16]`; user can pin via `cerberus.env.CERBERUS_PROJECT_KEY` in mala.yaml). Both fields are optional kwargs on `MalaConfig.__init__` and `from_env`; null defaults let the factory inject runtime-derived values without users having to set them. Update docstrings. |
| `src/orchestration/factory.py` | Exists | `_check_review_availability` / `_check_epic_verifier_availability` use `shutil.which("cerberus", ...)` + a `prompts/reviewers/` existence check on the resolved root; `_create_code_reviewer` / `_create_epic_verification_model` stop passing `bin_path`. No subcommand probe — v2's `spawn-epic-verify --help` could trigger a real verification run (extra dash-prefixed args are normalized as epic criteria). |
| `src/orchestration/config_resolution.py` | Exists | Verify nothing references `cerberus_bin_path`; YAML precedence unchanged. |
| `src/domain/validation/config_parser.py` | Exists | Docstring/wording updates only; YAML schema unchanged. |
| `src/domain/validation/config_types.py` | Exists | No change. |
| `src/domain/validation/e2e.py` | Exists | Replace cerberus preflight with PATH-based `cerberus` preflight. |
| `src/core/constants.py` | Exists | Audit for `cerberus` references; no behavior change expected. |
| `docs/project-config.md` | Exists | Reference `cerberus` v2 binary, env contract, removal of plugin discovery. |
| `docs/validation.md` | Exists | Update reviewer descriptions, binary name. |
| `docs/validation-triggers.md` | Exists | Update binary name + capability notes. |
| `docs/cli-reference.md` | Exists | Update env-var deprecation table and reviewer notes. |
| `CHANGELOG.md` | Exists or **New** | v2 prerequisite + removal of plugin auto-discovery + `cerberus_bin_path` removal. |
| `TODO.md` | Exists | Remove "Support cerberus v2" line. |
| `tests/unit/infra/test_cerberus_cli.py` | Renamed + rewritten | Legacy gate fixtures become `cerberus`; assertions on new flags / env / class name. |
| `tests/unit/infra/test_cerberus_output_parser.py` | Renamed + rewritten | v2 gate-state fixtures + iteration-findings fixtures. |
| `tests/unit/infra/test_cerberus_iteration_findings.py` | **New** | On-disk fixture tests against the real nested layout (`iterations/<N>/round-<R>/reviewers/<provider>#<i>/output.json`): happy multi-reviewer single-pass (`iterations/1/round-1/reviewers/{claude#1,codex#1,gemini#1}/output.json`); debate path with `round-1..round-3` asserting only `round-3` outputs are returned; iteration selection picks max of `iterations/1` and `iterations/2`; missing iterations dir; empty iteration dir; round dir with no `reviewers/`; reviewer dir missing `output.json`; malformed `output.json` skipped with parse error; P0–P3 priority parsing; line range edge cases. |
| `tests/unit/infra/test_cerberus_review.py` | Exists | Rewrite stub JSON to v2 shape; update spawn/wait commands; update env assertions. |
| `tests/unit/infra/clients/test_cerberus_epic_verifier.py` | Exists | Rewrite fake-subprocess JSON to v2; update env / arg assertions. |
| `tests/integration/orchestration/test_epic_verifier_config.py` | Exists | Update fake-binary fixtures and config expectations. |
| `tests/unit/orchestration/test_factory.py` | Exists | Update binary-name + availability-check assertions; cover the `cerberus spawn-epic-verify --help` probe. |
| `tests/unit/infra/test_config.py` | Exists | Delete `cerberus_bin_path` / plugin-discovery tests. |
| `tests/contracts/test_env_config_contract.py`, `tests/fakes/env_config.py` | Exists | Update for plugin discovery removal/rename. |
| `tests/fixtures/cerberus/*.json` | Exists | Replace v1 wait fixtures with v2 gate-state + per-reviewer JSON fixtures. |
| `tests/e2e/test_review_gate.py` | Exists | Rename/update to a real `cerberus` v2 smoke; skip when not on PATH. |
| `tests/integration/clients/test_cerberus_v2_smoke.py` | **New** | Golden-path integration test against a fake `cerberus` shell-script binary on per-test `PATH`. Cases: (a) verdict=pass, zero findings; (b) verdict=fail with 2 findings across 2 reviewers → `format_review_issues` renders both reviewer attributions; (c) stale-gate "already active" → `resolve` succeeds → second spawn proceeds; (d) verdict=fail with empty iteration dir → parse error. |

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes

- **`--max-rounds 0`**: v2 strictly rejects it; mala MUST stop passing it. Unit test asserts it is absent from spawn-code-review and spawn-epic-verify argv.
- **Empty or missing iteration / round / reviewers directory** (fail-closed): any of the following — missing `iterations/`, empty `iterations/`, missing `iterations/<N>/round-*`, or `round-<R>/reviewers/` containing no `output.json` files — is **always** surfaced as `ReviewResult(passed=False, parse_error="<which-layer-missing>", fatal_error=False)`, **regardless of the gate-state `verdict`**. A legitimate cerberus pass produces reviewer directories with `output.json` files whose `findings` array is empty; a missing reviewer directory is a different signal (cerberus crashed, disk error, layout drift, or a partial write) and MUST NOT be confused with success. parse_error text names the missing layer (e.g. `"no round-* directory under iterations/<N>"`, `"no output.json under round-<R>/reviewers/"`). Integration test (case d) asserts a `verdict=pass` gate-state with an empty reviewers directory still produces `passed=False` with a parse error.
- **Malformed per-reviewer JSON** (fail-closed): iteration-findings reader skips the bad file, records a parse error, and continues processing the other reviewers' files. `map_exit_code_to_result` **always** surfaces parse errors via `ReviewResult.parse_error`, even when other reviewers produced usable findings — in a consensus model a missing peer must invalidate the pass. On `verdict=pass` with a malformed peer, mala flips to `passed=False`. Unit test covers: (a) one malformed file + one valid file with findings → `passed=False` with parse error AND findings populated; (b) one malformed file + one valid file with empty findings → `passed=False` with parse error.
- **`verdict=requires_decision`**: mapped to `passed=False, fatal_error=False`. To prevent an empty-issues-list agent loop, the parser synthesizes a single `ReviewIssue` ("Cerberus reviewers reached no consensus — human decision or re-run required") so `format_review_issues` produces actionable follow-up text. Unit test asserts the synthetic issue is present and has priority `1` so the follow-up prompt surfaces it.
- **v2 binary not on `$PATH`**: `CerberusCLI.validate_binary()` returns `"cerberus binary not found in PATH"`; `factory._check_review_availability` returns the same; `reviewer_type: cerberus` is auto-disabled with that reason rather than crashing.
- **`CERBERUS_ROOT` cannot be resolved or points at a stripped install**: when `which("cerberus")` is set but the derived root has no `prompts/reviewers/` (e.g. binary copied to `~/bin/` without the support tree), `validate_binary()` returns `"cerberus root <path> missing prompts/reviewers/ — set cerberus.env.CERBERUS_ROOT in mala.yaml"` and `reviewer_type: cerberus` is auto-disabled. Unit test: simulate `which` returning a bin path whose `../prompts/reviewers/` is missing. Unit test: env-builder asserts `CERBERUS_ROOT` is present in the produced dict for every spawn/wait/resolve call.
- **Stale gate from a previous mala run** (no longer applicable in v2): v2 `StartSinglePass` warns-and-proceeds when `gate-state.json` is already pending (`cerberus/internal/orchestrator/orchestrator.go:90`), rather than rejecting the spawn the way v1 did. Combined with mala's per-attempt fresh `CERBERUS_RUN_KEY`, there is no `<state_root>/<project>/<run_key>/gate-state.json` collision in practice. mala therefore **removes** the v1 stale-gate spawn-resolve-retry path entirely. `CerberusCLI.resolve_gate(...)` stays in the CLI wrapper for user-facing `/cerberus:clear-gate` parity but is not called from mala's normal review path.
- **Spawn timeout**: propagates as `ReviewResult(passed=False, parse_error="spawn timeout", fatal_error=False)` (retryable).
- **`cerberus wait` exits non-zero with no JSON on stdout**: parser falls back to stderr; `map_exit_code_to_result` returns `parse_error` with the stderr tail, `fatal_error=False`.
- **Gate-state JSON has `status=pending` after wait returns**: treated as parse error (`"wait returned pending status"`) since wait blocks until resolved.
- **`current_iteration` missing or non-integer**: parser returns a nonfatal parse error; review retry policy decides.
- **`transcript_path` missing**: still parse verdict/findings; use computed run state dir as `review_log_path` fallback.
- **`CERBERUS_RUN_KEY` collision**: derived as `f"mala-{claude_session_id}"` (or `f"mala-{epic_session_token}"` for epic verification). mala generates a fresh session id per review/verify attempt, so collisions only happen on a session-id collision (vanishingly unlikely with `secrets.token_hex(6)` + Claude's UUID-based session ids). If observed, fall back to suffixing a monotonic timestamp.
- **Wait/status addressing mismatch**: mala MUST pass `--session-key $CERBERUS_RUN_KEY` (not `--session-id <unrelated_value>`) to `wait`/`status`. v2 resolves state at `<state_root>/<project_key>/<RunKey>/`; `--session-id X` clobbers RunKey to X and looks at the wrong path, returning `no_active_gate` for an actually-running gate. Asserted by the fake-binary smoke test.
- **`CERBERUS_STATE_ROOT` collision across concurrent mala runs**: state root rooted under `MALA_RUNS_DIR/<run_id>/cerberus` is per-run, so concurrent runs do not share directories.
- **`CERBERUS_PROJECT_KEY` derivation**: defaults to short SHA-256 of the absolute repo root path; **overridable** via `cerberus.env.CERBERUS_PROJECT_KEY` (the env-builder honors the override and the iteration-findings reader uses the final resolved value, so reads and writes agree). See OQ-Plan-2.
- **User passes `--max-rounds 0` (or other v1-only flags) via `spawn_args`/`wait_args`**: v2 rejects the spawn; mala surfaces the cerberus error verbatim. Documented in CHANGELOG.
- **v1 installs in `.claude/plugins/cache/cerberus/...`**: no longer auto-discovered. Migration callout in CHANGELOG.

#### Risk Register

| Risk | Impact | Mitigation / Test |
|------|--------|-------------------|
| Missing iteration / round / reviewers dir (any verdict) | Fail-open: gate-state `pass` could silently bypass review on a partial cerberus write | Always-parse-error rule (fail-closed regardless of verdict) + parser unit test + fake-binary integration test (case d: `verdict=pass` with empty reviewers dir → `passed=False`) |
| Partial malformed peer JSON in multi-reviewer run | Other reviewer's findings could mask the missing peer | Parse errors always surfaced via `ReviewResult.parse_error`; on `verdict=pass` with any peer parse error → `passed=False`. Unit tests for both populated-findings and empty-findings cases. |
| `requires_decision` produces empty-issues fix loop | Agent hallucinates fixes against no findings | Parser synthesizes one priority-1 `ReviewIssue` so follow-up prompt has actionable text; unit test asserts presence. |
| Malformed reviewer JSON | Bad review data silently dropped | Unit tests for invalid root, invalid `findings`, bad line fields |
| `requires_decision` mapping wrong | Inverted pass/fail behavior | OQ-Plan-3 resolved before merge; explicit unit-test assertion |
| `cerberus` not on PATH | Review silently disabled | Factory + CLI validation tests assert clear message |
| Stale gate from prior run | Not a real risk in v2 (warns-and-proceeds; fresh RUN_KEY per attempt avoids path collision) | Recovery code removed; regression test asserts `resolve` is NOT called during the normal review path |
| Project/run key collision | Wrong findings loaded | OQ-Plan-2 + state-path-isolation tests |
| cerberus reshuffles iteration layout | mala parser silently breaks | One centralized reader module so a future layout change is a one-file fix |
| Coverage drop | PR blocked by quality gate | Run coverage during PR; either add focused tests or resolve OQ-Plan-4 |

### Breaking Changes & Compatibility

- **mala.yaml**: no change to user-visible config shape.
- **CLI behavior**: a user-visible review run continues to look the same (mala blocks on the gate, prints results).
- **Binary prerequisite**: `cerberus` v2 must be on `$PATH`. Users with `.claude/plugins/cache/cerberus/cerberus/<1.x>/bin/` installs lose auto-discovery.
- **`MalaConfig.cerberus_bin_path` removed**: anyone constructing `MalaConfig` programmatically with this kwarg gets `TypeError` from the dataclass. No deprecation shim is added (per `CLAUDE.md` "No backward-compatibility shims"); this field was never a `mala.yaml` key and `MALA_CERBERUS_BIN_PATH` was never a read env var, so the kwarg path is the only user-facing surface. Documented in CHANGELOG.
- **Env vars**: the already-deprecated `MALA_CERBERUS_SPAWN_ARGS`/`WAIT_ARGS`/`ENV` and `MALA_REVIEW_TIMEOUT` continue to be parsed (no shape change, no new deprecations added by this PR). Their contents now flow to the v2 `cerberus` binary instead of the v1 gate wrapper, so v1-only flags (e.g. `--max-rounds 0`) embedded in those env vars or in `cerberus.env` will break. Documented in CHANGELOG; `docs/cli-reference.md`'s deprecation table is unchanged.
- **Import paths**: the retired client/parser import paths no longer exist (no shim per project policy).
- **Mitigations**: single mala minor-version bump with a CHANGELOG entry listing: v2 prerequisite, `cerberus_bin_path` removal, plugin auto-discovery removal, `--max-rounds 0` rejection, renamed modules.

## Testing & Validation Strategy

- **Unit tests**
  - `test_cerberus_cli.py`: binary name resolution; `build_env` produces all v2 env keys (`CERBERUS_HOST=generic`, `CERBERUS_ROOT`, run/state/project keys); `spawn_code_review`/`spawn_epic_verify` argv omits `--max-rounds 0`; `wait_for_review` constructs the expected v2 command, including `--session-key <run_key>` (NOT `--session-id`) so wait/status resolve `<state_root>/<project_key>/<run_key>/` — the same directory `spawn` populates via `CERBERUS_RUN_KEY`; `resolve_gate` unchanged.
  - `test_cerberus_output_parser.py`: `parse_gate_state` happy path, malformed JSON, missing fields, `verdict=null`/`pending` cases, exit-code-to-result mapping including non-zero / parse error / timeout. **Fail-closed assertions**: (a) `verdict=pass` + missing reviewers dir → `passed=False` with parse error; (b) `verdict=pass` + one malformed peer JSON + one valid peer with findings → `passed=False` with parse error AND findings populated; (c) `verdict=requires_decision` synthesizes one `ReviewIssue` with `priority=1` and the consensus-failure title so `format_review_issues` is non-empty.
  - `test_cerberus_iteration_findings.py` (**New**): happy single-pass (`iterations/1/round-1/reviewers/{claude#1,codex#1,gemini#1}/output.json` all parsed, reviewer attribution preserved from the directory name); debate run with `round-1`, `round-2`, `round-3` populated asserts only `round-3` findings are returned; latest-iteration auto-select across `iterations/1` and `iterations/2`; missing `iterations/` dir → `([], [parse_error])`; empty iteration dir → parse error; round dir with no `reviewers/` subdir → parse error; reviewer dir missing `output.json` skipped with parse error; malformed `output.json` skipped with parse error; P0–P3 priority parsing; line range edge cases.
  - `test_cerberus_review.py`: DefaultReviewer constructs the right CLI calls (no `--max-rounds 0`, correct env), short-circuits on no commits, surfaces no-changes spawn errors, recovers stale gate then re-spawns, returns parsed `ReviewResult` from a v2 stub.
  - `test_cerberus_epic_verifier.py`: same flow for epic verify; verdict mapping; per-reviewer findings → `UnmetCriterion`.
  - `test_factory.py`: `_check_review_availability` and `_check_epic_verifier_availability` return expected reasons when `cerberus` is not on PATH, when `CERBERUS_ROOT` resolves but lacks `prompts/reviewers/`, and `None` when both are present; `_create_code_reviewer` / `_create_epic_verification_model` no longer pass `bin_path`; assert that NO `cerberus` subprocess is invoked during availability checks (regression guard against the v2 `spawn-epic-verify --help` trap).
  - `test_config.py`: `MalaConfig.from_env` works without `cerberus_bin_path`; plugin-discovery tests removed.
- **Integration / end-to-end**
  - `tests/integration/clients/test_cerberus_v2_smoke.py` (**New**) — fake `cerberus` shell-script binary on a per-test `PATH` that:
    - on `spawn-code-review` / `spawn-epic-verify`: writes one `output.json` per reviewer at the **real v2 path** `<state_root>/<project_key>/<run_key>/iterations/1/round-1/reviewers/<provider>#<i>/output.json` (e.g. `claude#1`, `gemini#1`) and exits 0;
    - on `wait --json --session-key <K>`: asserts the value of `<K>` matches the spawn's `CERBERUS_RUN_KEY` (fail-loud if mala threads a different addressing key), reads gate-state from `<state_root>/<project_key>/<K>/`, prints v2 gate-state JSON on stdout (with `current_iteration=1`), and exits with the appropriate code;
    - on `resolve`: exits 0.
    Cases: (a) verdict=pass, single round, zero findings (round dir present with reviewer subdirs whose `output.json` has empty `findings`) → `ReviewResult(passed=True, issues=[])`; (b) verdict=fail, single round, 2 findings across 2 reviewers, asserting `format_review_issues` renders both `<provider>#<i>` reviewer attributions; (c) **regression: no resolve in normal path** — assert that across a normal pass/fail review, `resolve` is never invoked (guards against accidentally re-adding the dead v1 stale-gate retry); (d) **fail-closed on missing artifacts**: gate-state JSON says `verdict=pass` but `iterations/1/round-1/reviewers/` is empty → mala returns `passed=False` with the iteration-findings parse error, `fatal_error=False` (asserts the fail-closed rule regardless of gate-state verdict); (e) **debate fake**: writes `round-1` and `round-2` reviewer outputs with different findings, asserts mala surfaces only the `round-2` findings (final-round-wins behavior); (f) **requires_decision**: gate-state `verdict=requires_decision` with reviewers present → mala returns `passed=False` with exactly one synthetic `ReviewIssue` whose title and priority match the parser's hardcoded constant.
  - `test_epic_verifier_config.py`: exercises factory wiring with the same fake binary.
- **Regression**
  - `reviewer_type: agent_sdk` paths untouched — covered by existing factory + `AgentSDKReviewer` tests.
  - Cumulative review wiring still receives prior findings in follow-up prompts.
  - `rg` for stale v1 binary names, removed lookup helpers, and retired module paths returns only CHANGELOG history references.
- **Manual smoke**
  - Run mala against a real `cerberus` v2 binary on a small repo. Confirm:
    - cerberus is invoked at `$(which cerberus)`.
    - `CERBERUS_STATE_ROOT`/`CERBERUS_PROJECT_KEY`/`CERBERUS_RUN_KEY`/`CERBERUS_HOST=generic` are present in subprocess env.
    - On verdict=fail, mala's follow-up prompt contains finding title, body, file, and line range from the per-reviewer JSON.
    - On verdict=pass with zero findings, mala records review success.
  - Uninstall `cerberus` from `$PATH`; confirm the "cerberus binary not found in PATH" disable-reason and no crash.
- **Quality gates**
  - `uv sync` clean; `uvx ruff check .`; `uvx ruff format --check .`; `uvx ty check`; import-linter contracts; `uv run pytest -m unit`; `uv run pytest -m integration -n auto`.
  - Coverage at or above 85% (OQ-Plan-4 if temporarily dipped).
- **Observability**
  - Preserve `review_log_path` by using v2 `transcript_path` or the computed state run directory.
  - New log line: `Found N per-reviewer findings in iteration <path>` to make iteration-dir misses diagnosable.

### Acceptance Criteria Coverage

| Spec AC (derived) | Covered By |
|-------------------|------------|
| `reviewer_type: cerberus` invokes the v2 `cerberus` binary resolved via `$PATH` (never `cerberus`). | `CerberusCLI._cerberus_bin()`; unit + integration tests assert the invoked command starts with `cerberus`. |
| mala invokes `cerberus spawn-code-review` and `spawn-epic-verify` without `--max-rounds 0`. | `spawn_code_review` / `spawn_epic_verify` argv assertion in `test_cerberus_cli.py`. |
| All v2 env vars (`CERBERUS_HOST=generic`, `CERBERUS_RUN_KEY`, `CERBERUS_STATE_ROOT`, `CERBERUS_PROJECT_KEY`) are set for cerberus subprocesses. | Env-builder unit test + fake-binary integration env assertions. |
| `verdict=fail` + per-reviewer findings ⇒ `ReviewResult.passed=False` with populated `issues` (title/body/file/line range). | `cerberus_iteration_findings.read_findings` + `map_exit_code_to_result`; integration smoke (case b). |
| `verdict=pass` with zero findings ⇒ `ReviewResult.passed=True` with empty `issues` and no spurious parse error. | Parser branch + integration smoke (case a). |
| Findings flow into follow-up prompts via `format_review_issues`. | Fake-binary integration test + lifecycle/review runner regression tests. |
| Cerberus epic verification uses v2 spawn/wait and parses v2 artifacts. | Epic verifier unit tests + `test_epic_verifier_config.py`. |
| Missing binary ⇒ clear `cerberus binary not found in PATH` and `reviewer_type: cerberus` auto-disabled (no crash). | `factory._check_review_availability`; unit test in `test_factory.py`. |
| Availability check does NOT execute any `cerberus` subcommand (avoids `spawn-epic-verify --help` accidentally launching a verification). | `test_factory.py` regression test asserting no subprocess invocation. |
| `cerberus resolve --reason …` is NOT invoked on the normal review path (v2 warns-and-proceeds; v1 stale-gate retry removed). | Regression test in `test_cerberus_review.py` + integration smoke (case c). |
| `MalaConfig.from_env` has no `cerberus_bin_path` attribute and no plugin discovery call. | Field removal; unit tests in `test_config.py`. |
| No source/test references stale v1 binary names, removed lookup helpers, or retired module paths (except CHANGELOG). | `rg` sweep at end of PR. |
| `reviewer_type: agent_sdk` behavior unchanged. | `AgentSDKReviewer` paths untouched; existing factory tests cover. |
| Docs describe v2 binary and PATH requirement. | Docs grep + manual review of updated docs. |

### Validation / Acceptance Checklist

Before merging the PR:

- [ ] `uv sync` clean.
- [ ] `uvx ruff check .` and `uvx ruff format --check .` pass.
- [ ] `uvx ty check` passes.
- [ ] Import-linter contracts pass.
- [ ] `uv run pytest` passes (unit + integration).
- [ ] Coverage at or above resolved OQ-Plan-4 floor.
- [ ] `git grep` for stale v1 binary names, removed lookup helpers, and retired module paths returns only CHANGELOG history.
- [ ] Manual smoke against a real `cerberus` v2 binary: verdict=pass and verdict=fail flows both work end-to-end; follow-up prompts contain findings.
- [ ] Manual smoke with `cerberus` removed from `$PATH`: mala disables `reviewer_type: cerberus` with the expected reason, does not crash.
- [ ] CHANGELOG entry calls out: v2 prerequisite, `cerberus_bin_path` removal, plugin auto-discovery removal, `--max-rounds 0` rejection.
- [ ] PR description surfaces OQ-Plan-3 mapping and any Deviation Log entries for cerberus-team awareness.

## Spec/Legacy Fidelity

The cerberus v2 spec (`/Users/cyou/code/cerberus/docs/2026-05-08-rebuild-spec.md`) is the source of truth for the v2 CLI surface and env contract. Any deviation in mala (e.g. continuing to set `CLAUDE_SESSION_ID` alongside `CERBERUS_SESSION_ID`) is recorded in the Deviation Log below.

### Deviation Log

| Source | Deviation | Rationale | Approved? |
|--------|-----------|-----------|-----------|
| v2 env contract | mala sets `CLAUDE_SESSION_ID` only when the caller passes one (legacy compatibility), but no longer requires it. | v2 uses `CERBERUS_RUN_KEY` for state addressing; the v1-era `CLAUDE_SESSION_ID` requirement is removed. | Pending PR review. |
| v2 wait addressing | mala uses `cerberus wait --session-key $CERBERUS_RUN_KEY` instead of `--session-id`. | v2 keys state by RunKey; `--session-id` would override RunKey and look up the wrong directory under `<state_root>/<project>/`. Matches the cerberus generic-host hint in `spawn_code_review.go:123`. | Self-approved (correctness fix). |
| v2 verdict enum | mala maps `requires_decision` → `passed=False, fatal_error=False` (retryable). | `ReviewResult.passed` is boolean; least-surprising mapping. See OQ-Plan-3. | Pending confirmation from cerberus team — surface in PR description. |

## Open Questions

- **OQ-Plan-1 (per-reviewer JSON filename layout)**: **Resolved.** Verified against `cerberus/internal/state/paths.go` (`IterationDir`/`RoundDir`/`ReviewerDir`/`WriteReviewerOutput`) and a live run on disk. Layout is `<state_root>/<project>/<run>/iterations/<N>/round-<R>/reviewers/<provider>#<index>/output.json` with `<N>` and `<R>` 1-based. Single-pass runs produce exactly `round-1`; debate runs produce `round-1..round-maxRounds` and only the highest `round-N` is canonical. The iteration-findings reader encodes this traversal directly; no spike needed.
- **OQ-Plan-2 (project key derivation)**: **Resolved.** `CERBERUS_PROJECT_KEY = hashlib.sha256(os.path.abspath(repo_path).encode()).hexdigest()[:16]` (16-char hex prefix, ~64 bits — collision-free for any realistic number of mala projects per user). Overridable via `cerberus.env.CERBERUS_PROJECT_KEY` for cases where users want a stable cross-machine key. Documented in `docs/project-config.md`.
- **OQ-Plan-3 (`requires_decision` verdict mapping)**: **Resolved.** Map to `passed=False, fatal_error=False`. The parser synthesizes a single `ReviewIssue(title="Cerberus reviewers reached no consensus", priority=1)` so `format_review_issues` produces actionable follow-up text instead of an empty-issues fix loop. Documented in Deviation Log; explicit unit test in `test_cerberus_output_parser.py`.
- **OQ-Plan-4 (coverage floor)**: **Resolved.** Maintain the existing 85% line-coverage floor; if a commit in this PR temporarily dips below, add focused tests in the same PR rather than relax the floor. The PR description must note the final coverage delta. (No `mala.yaml` change required — threshold stays at 85.)

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
