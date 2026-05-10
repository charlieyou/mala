# Architecture Fixes Plan

Date: 2026-05-09

## Context

This plan decomposes the architecture-review follow-up work into four
workstreams. Workstream A covers orchestration refactors and is the normative
source for epic `mala-ctcy9`.

The workstream order is chained:

1. Workstream A - orchestration refactors.
2. Workstream B - infra/client refactors.
3. Workstream C - domain/config/factory refactors.
4. Workstream D - CLI boundary extraction.

## Goals

- Keep behavior observable-compatible while moving large orchestration logic
  behind smaller state machines and ports.
- Co-locate orchestration state machines in `src/pipeline/`.
- Replace callback-heavy boundaries with explicit protocol ports.
- Deduplicate trigger command resolution so validation execution and dry-run
  paths consume the same resolved plan.
- Keep all moves clean-break: update imports directly and do not add re-export
  shims.

## Non-Goals

- No broad rewrite of provider implementations in Workstream A.
- No CLI command-surface changes.
- No changes to user-visible validation semantics beyond preserving existing
  edge-case behavior through smaller state machines.
- No compatibility modules that only re-export moved symbols.

## Review Findings Covered

| Finding | Workstream | Summary |
| --- | --- | --- |
| #1 | A | `RunCoordinator` trigger validation state machine |
| #2 | A | `MalaOrchestrator` decoupling from callback/state wiring |
| #4 | A | `IssueExecutionCoordinator` extraction and queue behavior |
| #11 | A | Trigger command resolution deduplication |
| #3 | B | `codex_provider.py` split |
| #5 | B | Amp/Codex provider base helper |
| #6 | B | `codex_pre_tool_use.py` safety-critical split |
| #13 | B | `ClaudeAgentRuntimeBuilder` cleanup |
| #7 | C | Validation config parser extraction |
| #8 | C | Factory dependency assembly split |
| #12 | C | Config dataclass consolidation |
| #9 | D | CLI boundary extraction |

## Execution Model

Workstream A is a single workstream PR. Parallel agent execution may happen in
worktrees on tracking branch `workstream-a`; the owner cherry-picks green
sub-commit groups onto local main in the declared order below.

Each subtask should keep tests close to the behavior it changes. State-machine
surfaces land before coordinator adoption so later changes can be verified
against pure transition tests.

## Validation Baseline

Run the configured project checks before merging a workstream:

```bash
RUFF_CACHE_DIR=/tmp/ruff-${AGENT_ID:-default} uvx ruff format --check .
RUFF_CACHE_DIR=/tmp/ruff-${AGENT_ID:-default} uvx ruff check .
uvx ty check
uvx --from import-linter lint-imports
uv run pytest -o cache_dir=/tmp/pytest-${AGENT_ID:-default}
```

Focused checks listed under each task are required while developing that task.
The full validation baseline is required for the owner integration commit.

## Workstream A - Orchestration Refactors

### A.1 RunCoordinator State Machine (Finding #1)

**Problem.** `RunCoordinator._run_trigger_validation_loop` mixes trigger command
execution, failure-mode policy, remediation, code review, retry, terminal event
emission, and interrupt handling in one imperative loop. The mixed concerns make
interrupt precedence and terminal-state behavior hard to reason about.

**Target shape.**

- Add `src/pipeline/run_validation_state.py`.
- Keep the state machine pure: no I/O, no subprocess execution, no coordinator
  imports, and no event-sink calls.
- Mirror the naming and dataclass style used by `src/domain/lifecycle.py` and
  the pipeline lifecycle effect handling code.
- Drive coordinator behavior through explicit effects emitted by the
  transition function.

**New states.**

- `PROCESSING_COMMANDS`
- `AWAITING_REMEDIATION`
- `RUNNING_CODE_REVIEW`
- `REMEDIATING_REVIEW`
- `EMITTING_TERMINAL_EVENT`

**New event/effect surface.**

- Events describe facts observed by the coordinator: commands passed, command
  failed, remediation passed, remediation failed, code review passed, code
  review produced findings, retry exhausted, and interrupt requested.
- Effects describe work for the coordinator to perform: run commands, start
  remediation, start code review, emit terminal success, emit terminal failure,
  and record run validation state.
- Terminal effects must only be emitted from `EMITTING_TERMINAL_EVENT`.

**Task A.1.1 - integration-path state-machine skeleton.**

- Primary files:
  - `src/pipeline/run_validation_state.py` (new)
  - `tests/unit/pipeline/test_run_validation_state.py` (new)
- Goal: land state/event/effect enums, frozen dataclasses, and a pure
  `transition()` function with a full state-by-event matrix.
- Scope:
  - In: state machine types, explicit transition table, rejection behavior for
    invalid state/event pairs, terminal effects.
  - Out: no `run_coordinator.py` changes in this task.
- Acceptance criteria:
  - Every state/event pair has an explicit assertion of next state plus emitted
    effects or rejection.
  - `transition()` is pure.
  - Terminal effects only come from `EMITTING_TERMINAL_EVENT`.
- Verification:
  - `uv run pytest tests/unit/pipeline/test_run_validation_state.py -v`
  - `uvx ruff check src/pipeline/run_validation_state.py tests/unit/pipeline/test_run_validation_state.py`
  - `uvx ty check src/pipeline/run_validation_state.py`

**Task A.1.2 - drive trigger validation through the state machine.**

- Primary files:
  - `src/pipeline/run_coordinator.py`
  - `tests/unit/pipeline/test_trigger_execution.py`
  - `tests/unit/pipeline/test_run_validation_state.py`
- Goal: replace the ad hoc trigger validation loop state branching with the
  pure transition surface from A.1.1.
- Scope:
  - In: map command results, review outcomes, remediation outcomes, and
    interrupts into state-machine events; execute emitted effects in the
    coordinator.
  - Out: no trigger command resolver extraction yet; that is A.4.
- Acceptance criteria:
  - Existing trigger validation behavior is preserved.
  - Interrupts take precedence over retry/remediation continuation.
  - Run validation failure context is preserved when the terminal state is
    failure.
  - Run validation terminal events are emitted after the state machine reaches a
    terminal-emission state, not before.
- Verification:
  - `uv run pytest tests/unit/pipeline/test_trigger_execution.py tests/unit/pipeline/test_run_validation_state.py -v`

### A.2 Protocol-Port Introduction (Finding #2)

**Problem.** `MalaOrchestrator` and adjacent coordinator code exchange behavior
through callback references and concrete client assumptions. This makes callback
migration fragile and leaves orchestration code coupled to lifecycle mutation
details.

**Target shape.**

- Keep protocol definitions in `src/core/protocols/` with focused modules.
- Use the existing `IssueProvider` and `MalaEventSink` ports.
- Add `IssueLifecyclePort` as the explicit mutation boundary for per-issue
  lifecycle state.
- Replace issue finalizer and epic callback references with protocol methods.
- Do not create re-export shims. Update imports directly.

**Required port surface.**

- `IssueProvider` remains the issue storage/scheduling provider. Extend it only
  for operations actually needed by migrated callbacks.
- `MalaEventSink` remains the event/output sink. Do not bypass it from pipeline
  orchestration code.
- `IssueLifecyclePort` exposes lifecycle snapshot/mutation behavior needed by
  finalization, abort, and interrupt handling.

**Task A.2.1 - add issue lifecycle port contract.**

- Primary files:
  - `src/core/protocols/issue_lifecycle_port.py` (new)
  - `tests/contracts/test_issue_lifecycle_port_contract.py` (new)
  - lifecycle fakes used by coordinator tests
- Goal: define a protocol and small dataclass request/snapshot shapes for
  lifecycle mutation effects.
- Scope:
  - In: port protocol, frozen dataclasses, idempotent fake implementation for
    tests.
  - Out: no coordinator migration yet.
- Acceptance criteria:
  - Contract tests assert the dataclass shapes and protocol methods.
  - Fake abort/interrupt handling is idempotent.
  - Dataclass decorators are present and guarded by tests.
- Verification:
  - `uv run pytest tests/contracts/test_issue_lifecycle_port_contract.py -v`

**Task A.2.2 - migrate issue finalizer callbacks to ports.**

- Primary files:
  - `src/pipeline/issue_finalizer.py`
  - `src/pipeline/lifecycle_effect_handler.py`
  - finalizer/coordinator tests
- Goal: replace issue finalizer callback references with
  `IssueLifecyclePort`, `IssueProvider`, and `MalaEventSink`.
- Acceptance criteria:
  - Finalizer code depends on protocol ports, not callback bundles.
  - Existing finalization and abort behavior is preserved.
  - Interrupt event exposure is covered by tests.
- Verification:
  - `uv run pytest tests/unit/pipeline/test_lifecycle_effect_handler.py tests/contracts/test_issue_lifecycle_port_contract.py -v`

**Task A.2.3 - migrate epic callback references.**

- Primary files:
  - `src/infra/epic_verifier.py`
  - `src/core/protocols/issue.py`
  - epic verification tests
- Goal: extend `IssueProvider` for the specific epic callback result paths and
  migrate callers to the port.
- Acceptance criteria:
  - Epic callback references are removed from orchestration wiring.
  - `IssueProvider` exposes the needed predicate/result behavior.
  - Existing epic verification tests pass.
- Verification:
  - `uv run pytest tests/contracts/test_issue_provider_contract.py tests/unit/infra/test_epic_verifier.py -v`

**Task A.2.4 - remove guarded delegation wiring.**

- Primary files:
  - `src/orchestration/orchestrator.py`
  - `tests/unit/orchestration/test_orchestrator.py`
- Goal: after the protocol ports are in place, remove guarded callback
  delegation branches that only existed for mixed old/new wiring.
- Acceptance criteria:
  - Orchestrator construction uses explicit ports.
  - No old callback fallback remains for migrated behavior.
  - Unit tests cover the new construction path.
- Verification:
  - `uv run pytest tests/unit/orchestration/test_orchestrator.py -v`

### A.3 IssueExecutionCoordinator Extraction (Finding #4)

**Problem.** Issue scheduling and per-issue execution behavior was split across
orchestrator state, active task tracking, and provider polling logic. Capacity
and retry behavior were especially difficult to validate after failures and
terminal polls.

**Target shape.**

- `IssueExecutionCoordinator` owns run-loop scheduling and active task tracking.
- Work-queue decision logic lives in `src/pipeline/work_queue.py`.
- Provider polling failures are represented as typed decisions and retried when
  transient.
- Interrupt handling is ordered: stop spawning before draining or aborting, and
  preserve terminal poll semantics.

**Task A.3.1 - add work queue decisions.**

- Primary files:
  - `src/pipeline/work_queue.py` (new)
  - `tests/unit/pipeline/test_work_queue.py` (new)
- Goal: introduce a typed work queue surface for provider polling decisions.
- Acceptance criteria:
  - Ready, empty, terminal-drain, transient-error, and capacity decisions are
    represented explicitly.
  - Active ready capacity is not counted as available capacity.
  - Terminal poll retry waits are skipped.
- Verification:
  - `uv run pytest tests/unit/pipeline/test_work_queue.py -v`

**Task A.3.2 - use work queue in issue coordinator.**

- Primary files:
  - `src/pipeline/issue_execution_coordinator.py`
  - `tests/unit/pipeline/test_issue_execution_coordinator.py`
  - `tests/integration/pipeline/test_watch_mode.py`
- Goal: route coordinator polling and scheduling through the typed work queue.
- Acceptance criteria:
  - Spawn capacity is preserved after task failures.
  - Terminal poll drain is signaled and tested.
  - Transient poll failures are retried without losing active work.
  - Coordinator interrupt ordering is preserved.
- Verification:
  - `uv run pytest tests/unit/pipeline/test_issue_execution_coordinator.py tests/unit/pipeline/test_work_queue.py -v`

**Task A.3.3 - record validation and interrupt edge cases.**

- Primary files:
  - `src/pipeline/issue_execution_coordinator.py`
  - `src/pipeline/run_coordinator.py`
  - focused coordinator tests
- Goal: preserve edge-case behavior around remediation interrupts and run
  validation records.
- Acceptance criteria:
  - Remediation interrupts record run validation state before aborting.
  - Interrupt precedence is preserved in remediation paths.
  - Failure context survives terminal state transitions.
- Verification:
  - `uv run pytest tests/unit/pipeline/test_issue_execution_coordinator.py tests/unit/pipeline/test_trigger_execution.py -v`

### A.4 Trigger Command Resolution Deduplication (Finding #11)

**Problem.** Trigger command resolution existed in multiple places, making
runtime execution, dry-run behavior, and tests vulnerable to drifting command
override semantics.

**Target shape.**

- Add `src/pipeline/trigger_plan.py`.
- Keep command ref resolution pure and shared.
- `RunCoordinator` and `TriggerEngine` consume the same resolved command plan.
- Preserve strict startup config validation for invalid refs.

**Task A.4.1 - extract trigger command resolution.**

- Primary files:
  - `src/pipeline/trigger_plan.py` (new)
  - `src/pipeline/run_coordinator.py`
  - `src/pipeline/trigger_engine.py`
  - `tests/unit/pipeline/test_trigger_plan.py` (new)
  - `tests/unit/pipeline/test_trigger_execution.py`
- Goal: move command ref resolution and override application to a shared pure
  helper.
- Acceptance criteria:
  - Ref, command override, timeout override, and empty command-list behavior are
    identical for runtime and dry-run consumers.
  - Missing refs still fail fast through config/spec validation.
  - Runtime exceptions are not the primary unknown-ref validation mechanism.
- Verification:
  - `uv run pytest tests/unit/pipeline/test_trigger_plan.py tests/unit/pipeline/test_trigger_execution.py -v`

### A.5 Integration Order

Owner cherry-pick order for Workstream A:

1. A.1.1 state-machine skeleton and matrix tests.
2. A.1.2 coordinator adoption of run validation state machine.
3. A.2.1 lifecycle port contract and fakes.
4. A.2.2 issue finalizer callback migration.
5. A.2.3 epic callback migration and `IssueProvider` extension.
6. A.2.4 removal of guarded delegation wiring.
7. A.3.1 work queue decisions.
8. A.3.2 issue coordinator work queue adoption.
9. A.3.3 interrupt/validation edge-case fixes.
10. A.4.1 trigger command resolution extraction.

### A.6 Workstream A Acceptance Criteria

- `RunCoordinator` trigger validation is driven by a pure state machine in
  `src/pipeline/run_validation_state.py`.
- State machines introduced by this workstream are co-located in
  `src/pipeline/`.
- `MalaOrchestrator` delegates through explicit protocol ports and no migrated
  callback fallback remains.
- `IssueLifecyclePort` exists and is covered by contract tests.
- `IssueProvider`, `MalaEventSink`, and `IssueLifecyclePort` form the protocol
  boundary for migrated orchestration behavior.
- `IssueExecutionCoordinator` owns scheduling decisions through
  `src/pipeline/work_queue.py`.
- Trigger command resolution is centralized in `src/pipeline/trigger_plan.py`.
- Existing trigger, lifecycle, coordinator, and epic verification tests pass.

## Workstream B - Infra/Clients Refactors

Workstream B addresses findings #3, #5, #6, and #13. It is sequenced after
Workstream A. `codex_provider.py` should become a facade over extracted
selftest, MCP factory, and plugin installer helpers. Amp and Codex provider
shared plugin installation should use a Protocol plus helper module, not an
ABC. The `codex_pre_tool_use.py` split is safety-critical and requires a
golden byte-identical corpus before extraction. `ClaudeAgentRuntimeBuilder`
should accept preconstructed hooks/caches and format them into SDK structures.

## Workstream C - Domain/Config/Factory Refactors

Workstream C addresses findings #7, #8, and #12. It is sequenced after
Workstream B. Validation config parsing moves to a dedicated parser so
`ValidationConfig` remains a pure dataclass boundary. Runtime dependency
assembly gets named dataclasses instead of positional tuples. Config bundle
overlap is reduced by deriving narrow runner views from `PipelineConfig`.

## Workstream D - CLI Boundary Extraction

Workstream D addresses finding #9. It is sequenced after Workstream C. The CLI
should become argument adapter plus output formatting only. Move CLI override
application, dry-run handling, validators, scope parsing, and init YAML
generation into typed orchestration/domain helpers while preserving flags, exit
codes, dry-run behavior, and init output.

## Shared Test Strategy

### State-Machine Tests

Every pure state machine introduced by this plan gets a state-by-event matrix
test. Invalid transitions must be asserted, not left implicit.

### Golden Tests

Safety-critical text/output preserving refactors, especially
`codex_pre_tool_use.py`, must land a golden byte-identical corpus before moving
policy code.

### Contract Tests

New protocol ports require contract tests that cover dataclass shapes and fake
implementations used by unit/integration tests.

## Open Questions

- If a runner-specific config dataclass has genuinely unique fields, keep it as
  a narrow view rather than forcing unrelated fields into `PipelineConfig`.
- If selftest marker behavior in `codex_pre_tool_use.py` is not cleanly
  separable, document why in the relevant commit and leave adapter behavior
  unchanged.
