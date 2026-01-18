# Implementation Plan: Epic Verification Reviewer Choice

## Context & Goals
- **Spec**: `docs/specs/2026-01-18-epic-verification-reviewer-choice.md`
- Enable users to choose between Cerberus or Agent SDK for epic verification via `reviewer_type` config
- Mirror the existing code review pattern (`per_issue_review.reviewer_type`)
- Default to `agent_sdk` for backwards compatibility

## Scope & Non-Goals
- **In Scope**
  - R1: New `epic_verification` config section with `reviewer_type`
  - R2: Config schema parity with code review (enabled, timeout, max_retries, failure_mode, cerberus block)
  - R3: Protocol abstraction with Cerberus and Agent SDK implementations
  - R4: Cerberus implementation using `spawn-epic-verify` command
  - R5: Availability checking with explicit failure for Cerberus
  - R6: Retry policy per failure category (error typing for retryable vs non-retryable)
  - R7: Event observability with `reviewer_type` field

- **Out of Scope (Non-Goals)**
  - Changing epic verification semantics (acceptance criteria matching, remediation logic)
  - Adding new verification capabilities beyond reviewer selection
  - Deprecating Agent SDK option
  - New event types (only extend existing events with `reviewer_type`)
  - Implementation of `spawn-epic-verify` Cerberus command (external dependency)

## Assumptions & Constraints
- Cerberus `spawn-epic-verify` command follows the same spawn/wait pattern as `review-gate`
- `review-gate wait --json` output can be mapped into `EpicVerdict`
- The existing `EpicVerificationModel` protocol is the right abstraction layer for swapping implementations
- Must not break existing default behavior (`agent_sdk`)

### Implementation Constraints
- Extend `ValidationConfig` to include `epic_verification` section (new `EpicVerificationConfig` dataclass)
- Use existing `EpicVerificationModel` protocol for new implementations
- Mirror `CodeReviewConfig` field structure for consistency
- Use existing `CerberusConfig` dataclass pattern for nested Cerberus settings
- Use existing `find_cerberus_bin_path()` pattern for binary location
- Cerberus subcommand detection: Run `spawn-epic-verify --help` and require exit 0 (per spec R5)

### Testing Constraints
- Mock Cerberus subprocess for unit tests (no real subprocess in CI)
- No integration tests with real Cerberus until plugin ships `spawn-epic-verify`
- Event emission tests to verify `reviewer_type` field propagation
- Must maintain existing test coverage for agent_sdk path

## Integration Analysis

### Existing Mechanisms Considered

| Existing Mechanism | Could Serve Feature? | Decision | Rationale |
|--------------------|---------------------|----------|-----------|
| `CodeReviewConfig` in config.py (L252-278) | Pattern only | Create parallel `EpicVerificationConfig` | Same field structure, different config section |
| `_create_code_reviewer()` in factory.py (L445-519) | Pattern only | Create `_create_epic_verification_model()` | Mirrors reviewer selection logic |
| `_check_review_availability()` in factory.py (L376-442) | Pattern only | Create `_check_epic_verifier_availability()` | Same binary/subcommand validation pattern |
| `EpicVerificationModel` protocol | Yes | Extend with Cerberus impl | Already model-agnostic, just add implementation |
| `ClaudeEpicVerificationModel` | Yes | Keep as Agent SDK impl | Existing code becomes one of two options |
| `DefaultReviewer` spawn-wait pattern (cerberus_review.py) | Pattern only | Create `CerberusEpicVerifier` | Mirrors subprocess orchestration |
| `EpicVerificationCoordinator` | Partial | Keep retry loop | Already has retry logic via `max_retries`; receives model via callbacks |
| `EpicVerifier` | Yes | Modify | Add `reviewer_type` to event emissions |

### Integration Approach
1. **Config layer**: Add `EpicVerificationConfig` dataclass to `config.py`, parser to `config_loader.py`
2. **Factory layer**: Add `_create_epic_verification_model()` with selection logic, `_check_epic_verifier_availability()` with subcommand probe
3. **Implementation layer**: Create `CerberusEpicVerifier` mirroring `DefaultReviewer` spawn-wait pattern
4. **Orchestration layer**: `EpicVerifier` receives model via constructor (already protocol-based); pass `reviewer_type` for events
5. **Events layer**: Add `reviewer_type` parameter to existing epic verification events in `DiagnosticsEvents`

## Prerequisites
- [x] Cerberus plugin architecture supports adding new subcommands (assumed)
- [ ] Cerberus plugin `spawn-epic-verify` command exists (external dependency - can proceed without, will be disabled)
- [ ] No additional access or approvals required

## High-Level Approach

**Phasing: Layers approach** (config -> implementation -> wiring -> events)

### Phase 1: Configuration Layer
1. Add `EpicVerificationConfig` dataclass to `src/domain/validation/config.py`
   - Fields: `enabled`, `reviewer_type`, `timeout`, `max_retries`, `failure_mode`, `cerberus`, `agent_sdk_timeout`, `agent_sdk_model`
2. Add `epic_verification: EpicVerificationConfig | None` field to `ValidationConfig`
3. Add `_parse_epic_verification_config()` to `src/domain/validation/config_loader.py`
4. Add unit tests for config parsing (defaults, valid configs, invalid reviewer_type)

### Phase 2: Implementation Layer
1. Create `src/infra/clients/cerberus_epic_verifier.py` with `CerberusEpicVerifier` class
   - Implement `EpicVerificationModel` protocol
   - Spawn-wait pattern: `spawn-epic-verify <epic-file> [diff args]` then `wait --json`
   - Map review-gate wait JSON to `EpicVerdict`
   - Error classification for R6 categories (timeout, parse_error, execution_error)
2. Add unit tests with mocked subprocess calls

### Phase 3: Wiring Layer
1. Add `_check_epic_verifier_availability()` to `src/orchestration/factory.py`
   - For `agent_sdk`: always available
   - For `cerberus`: probe `spawn-epic-verify --help`, require exit 0
2. Add `_create_epic_verification_model()` factory function
   - Select implementation based on `reviewer_type` config
   - Pass config-specific settings (timeout, model, cerberus args)
3. Update `_build_orchestrator_dependencies()` to use factory
4. Store `reviewer_type` for event emission

### Phase 4: Events Layer
1. Update `DiagnosticsEvents` protocol in `src/core/protocols/events/diagnostics.py`:
   - `on_epic_verification_started(epic_id: str, reviewer_type: str)`
   - `on_epic_verification_passed(epic_id: str, confidence: float, reviewer_type: str)`
   - `on_epic_verification_failed(..., reviewer_type: str)`
2. Update all event sink implementations:
   - `src/infra/io/base_sink.py`
   - `src/infra/io/console_sink.py`
   - `tests/fakes/event_sink.py`
3. Update `EpicVerifier` to pass `reviewer_type` to event emissions
4. Add event emission tests

## Technical Design

### Architecture

**Config Flow:**
```
mala.yaml
  └─> config_loader.py (_parse_epic_verification_config)
        └─> EpicVerificationConfig
              └─> factory.py (_create_epic_verification_model)
                    └─> CerberusEpicVerifier | ClaudeEpicVerificationModel
```

**Verification Flow (Cerberus):**
```
EpicVerifier.verify_and_close_epic()
  └─> emit on_epic_verification_started(epic_id, "cerberus")
  └─> CerberusEpicVerifier.verify()
        └─> write epic markdown to temp file
        └─> spawn: spawn-epic-verify <epic-file> [diff args]
        └─> wait: review-gate wait --json
        └─> map wait JSON → EpicVerdict
  └─> emit on_epic_verification_passed/failed(..., "cerberus")
```

**Verification Flow (Agent SDK):**
```
EpicVerifier.verify_and_close_epic()
  └─> emit on_epic_verification_started(epic_id, "agent_sdk")
  └─> ClaudeEpicVerificationModel.verify()
        └─> Claude SDK agent with read-only tools
        └─> parse response → EpicVerdict
  └─> emit on_epic_verification_passed/failed(..., "agent_sdk")
```

### Data Model

**New `EpicVerificationConfig` dataclass** (mirrors `CodeReviewConfig` L252-278):

```python
@dataclass(frozen=True)
class EpicVerificationConfig:
    """Configuration for epic verification reviewer selection.

    Attributes:
        enabled: Whether epic verification is enabled.
        reviewer_type: Type of reviewer to use ('cerberus' or 'agent_sdk').
        timeout: Top-level timeout in seconds (default: 600).
        max_retries: Maximum retry attempts on failure (default: 3).
        failure_mode: How to handle verification failures.
        cerberus: Cerberus-specific settings (timeout, spawn_args, wait_args, env).
        agent_sdk_timeout: Timeout in seconds for Agent SDK verification.
        agent_sdk_model: Model for Agent SDK verifier ('sonnet', 'opus', 'haiku').
    """

    enabled: bool = True
    reviewer_type: Literal["cerberus", "agent_sdk"] = "agent_sdk"
    timeout: int = 600
    max_retries: int = 3
    failure_mode: FailureMode = FailureMode.CONTINUE
    cerberus: CerberusConfig | None = None
    agent_sdk_timeout: int = 600
    agent_sdk_model: Literal["sonnet", "opus", "haiku"] = "sonnet"
```

**Cerberus Epic File Input** (passed to `spawn-epic-verify`):
- Write the epic description (including acceptance criteria) to a temp markdown file.
- Pass `spawn-epic-verify <epic-file>` plus diff args (`--commit <sha...>` preferred; range or `--uncommitted` fallback).
- Use a generated `CLAUDE_SESSION_ID` prefixed by epic ID for gate scoping.

**Error Categories (R6)** - Exception types for retry policy:
- `CerberusUnavailableError` - binary/subcommand missing (non-retryable)
- `ConfigError` - invalid config (non-retryable)
- `TimeoutError` - subprocess timeout (retryable)
- `ExecutionError` - subprocess non-zero exit (retryable)
- `ParseError` - invalid JSON response (retryable)
- `VerdictFailError` - verification failed with unmet criteria (not an error, normal flow)

### API/Interface Design

**EpicVerificationModel Protocol** (existing, unchanged):
```python
class EpicVerificationModel(Protocol):
    async def verify(
        self,
        epic_criteria: str,
        commit_range: str,
        commit_list: str,
        spec_content: str | None,
    ) -> EpicVerdictProtocol:
        ...
```

**CerberusEpicVerifier** (new, implements `EpicVerificationModel`):
```python
class CerberusEpicVerifier:
    def __init__(
        self,
        repo_path: Path,
        bin_path: Path | None,
        spawn_args: tuple[str, ...],
        wait_args: tuple[str, ...],
        env: dict[str, str],
        event_sink: MalaEventSink,
    ) -> None: ...

    async def verify(
        self,
        epic_criteria: str,
        commit_range: str,
        commit_list: str,
        spec_content: str | None,
        *,
        epic_id: str = "",
    ) -> EpicVerdictProtocol: ...
```

**Updated Event Signatures** (add `reviewer_type` parameter):
```python
def on_epic_verification_started(self, epic_id: str, reviewer_type: str = "agent_sdk") -> None: ...
def on_epic_verification_passed(self, epic_id: str, confidence: float, reviewer_type: str = "agent_sdk") -> None: ...
def on_epic_verification_failed(
    self,
    epic_id: str,
    unmet_count: int,
    remediation_ids: list[str],
    *,
    reason: str | None = None,
    reviewer_type: str = "agent_sdk",
) -> None: ...
```

Note: Default `reviewer_type="agent_sdk"` preserves backward compatibility for existing callers.

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/domain/validation/config.py` | Modify | Add `EpicVerificationConfig` dataclass |
| `src/domain/validation/config_loader.py` | Modify | Add `_parse_epic_verification_config()` function |
| `src/orchestration/factory.py` | Modify | Add `_create_epic_verification_model()`, `_check_epic_verifier_availability()` |
| `src/infra/epic_verifier.py` | Modify | Pass `reviewer_type` to event emissions |
| `src/core/protocols/events/diagnostics.py` | Modify | Add `reviewer_type` parameter to epic verification event signatures |
| `src/infra/io/base_sink.py` | Modify | Update event handler signatures |
| `src/infra/io/console_sink.py` | Modify | Update event handler signatures |
| `tests/fakes/event_sink.py` | Modify | Update fake event handlers |
| `src/infra/clients/cerberus_epic_verifier.py` | **New** | `CerberusEpicVerifier` implementation |
| `tests/unit/domain/validation/test_epic_verification_config.py` | **New** | Config parsing tests |
| `tests/unit/infra/clients/test_cerberus_epic_verifier.py` | **New** | Unit tests for Cerberus verifier |

## Risks, Edge Cases & Breaking Changes

### Risks
- **Cerberus plugin not updated**: If `spawn-epic-verify` command doesn't exist in the plugin, users configuring `reviewer_type: cerberus` get `cerberus_unavailable` error at startup. Mitigation: Clear error message with instructions.
- **Parse error from Cerberus**: Malformed JSON response from subprocess. Mitigation: Graceful handling with retry (R6 `parse_error` category).
- **Subprocess timeout**: Long-running verification exceeds timeout. Mitigation: Configurable timeout, retryable error category.

### Edge Cases & Failure Modes
- **Cerberus binary exists but subcommand missing**: `spawn-epic-verify --help` returns non-zero. Fail with actionable error: "cerberus plugin detected but spawn-epic-verify subcommand unavailable".
- **Timeout during verification**: Classify as `timeout` error, retryable per R6.
- **Empty commit list**: Already handled in existing code (evaluates criteria without code context).
- **Spec files exceed size limit**: Truncate with warning, pass truncated content to verifier.
- **Config missing entirely**: Use defaults (`enabled: true`, `reviewer_type: agent_sdk`).
- **Invalid reviewer_type value**: Fail config validation at startup with clear error.

### Breaking Changes & Compatibility
- **Event signature changes**: `on_epic_verification_started`, `on_epic_verification_passed`, `on_epic_verification_failed` gain `reviewer_type` parameter. Using default value (`"agent_sdk"`) ensures backward compatibility for existing call sites.
- **No breaking changes to**:
  - `EpicVerdict` schema (unchanged)
  - `EpicVerificationModel` protocol (unchanged)
  - Remediation logic (unchanged)
  - `validation_triggers.epic_completion` behavior (still controls **when** verification runs)

## Testing & Validation Strategy

### Unit Tests
- **Config parsing** (`tests/unit/domain/validation/test_epic_verification_config.py`):
  - Parse valid config with all fields
  - Verify defaults when section omitted
  - Verify defaults when fields omitted
  - Reject invalid `reviewer_type` value
  - Reject unknown fields (strict parsing)

- **CerberusEpicVerifier** (`tests/unit/infra/clients/test_cerberus_epic_verifier.py`):
  - Mock subprocess calls, verify command construction
  - Parse valid wait JSON response to EpicVerdict
  - Handle timeout → raise TimeoutError
  - Handle non-zero exit → raise ExecutionError
  - Handle invalid JSON → raise ParseError
  - Verify epic file is written and cleaned up
  - Verify CLAUDE_SESSION_ID is generated with epic prefix

- **Factory selection**:
  - `reviewer_type: agent_sdk` → `ClaudeEpicVerificationModel`
  - `reviewer_type: cerberus` (available) → `CerberusEpicVerifier`
  - `reviewer_type: cerberus` (unavailable) → error/disabled

- **Event emission**:
  - Verify `reviewer_type` field included in all epic verification events
  - Verify default value works for existing call sites

### Integration Tests
- No real Cerberus subprocess tests until plugin ships
- E2E test with `reviewer_type: agent_sdk` (existing coverage suffices)

### Manual Validation
1. Configure `epic_verification.reviewer_type: cerberus` in mala.yaml
2. Run with Cerberus plugin installed → verify `spawn-epic-verify` invoked correctly
3. Run without Cerberus plugin → verify actionable error message at startup
4. Configure `reviewer_type: agent_sdk` → verify existing behavior preserved
5. Omit `epic_verification` section → verify defaults applied

### Acceptance Criteria Coverage

| Spec AC | Covered By |
|---------|------------|
| R1: Config section with `reviewer_type` | `EpicVerificationConfig` dataclass, `_parse_epic_verification_config()` |
| R2: Schema parity with code review | `EpicVerificationConfig` fields mirror `CodeReviewConfig` |
| R3: Protocol abstraction | Use existing `EpicVerificationModel`, add `CerberusEpicVerifier` impl |
| R4: Cerberus spawn-wait command | `CerberusEpicVerifier` implementation |
| R5: Availability check | `_check_epic_verifier_availability()` with subcommand probe |
| R6: Retry policy per failure category | Error exception types, coordinator retry loop handles |
| R7: Event observability | `reviewer_type` parameter on epic verification events |

## Spec/Legacy Fidelity
- Plan aligns with spec requirements R1-R7
- Default `reviewer_type: agent_sdk` preserves existing behavior

### Deviation Log
| Source | Deviation | Rationale | Approved? |
|--------|-----------|-----------|-----------|
| None | — | — | — |

## Open Questions

1. ~~**Cerberus subcommand detection**~~ → RESOLVED: Run `spawn-epic-verify --help` and require exit 0 (per spec R5)

2. ~~**Agent SDK config fields**~~ → RESOLVED: Include `agent_sdk_timeout` and `agent_sdk_model` (mirror CodeReviewConfig)

3. ~~**Test strategy**~~ → RESOLVED: Mock subprocess only; no real Cerberus tests until plugin ships

4. ~~**Phasing**~~ → RESOLVED: Layers approach (config → impl → wiring → events)

5. **Retry loop ownership (R6)**: The existing `EpicVerificationCoordinator` has retry logic via callbacks. The plan proposes:
   - `CerberusEpicVerifier` classifies errors into exception types (TimeoutError, ExecutionError, ParseError)
   - Existing coordinator retry loop catches these and decides retry vs fail based on `max_retries`
   - Non-retryable errors (CerberusUnavailableError, ConfigError) bypass retry loop
   - This approach requires no changes to `EpicVerificationCoordinator` beyond receiving the model via factory

## Next Steps
After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
