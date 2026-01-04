# Implementation Plan: Architecture Refactoring

## Context & Goals

- **Source**: Architecture review identified 10 issues (3 High, 5 Medium, 2 Low priority)
- **Primary Goal**: Both testability AND maintainability equally prioritized
- **Success Metrics**:
  - `AgentSessionRunner` delegates all stream processing to `MessageStreamProcessor`; no SDK iteration calls remain
  - `ContextPressureHandler` exposes checkpoint/restart logic independently testable without `AgentSessionRunner`
  - `DefaultReviewer.__call__` orchestrates only; all subprocess and parsing logic in dedicated components
  - `_cli_main` only parses/dispatches; command handlers perform all lock operations
  - Import-linter passes with new SDK boundary rules
  - All new components have unit tests
- **Phasing**: Issue-by-issue (complete each end-to-end before starting next)

## Scope & Non-Goals

### In Scope (8 Issues)

**HIGH PRIORITY (3)**
1. SDK imports in pipeline - Move to infra adapter
2. Duplicated agent runtime setup - Create AgentRuntimeBuilder
3. DefaultReviewer complexity - Split into CLI/Parser/Adapter

**MEDIUM PRIORITY (5)**
4. AgentSessionRunner god class - Extract MessageStreamProcessor, ContextPressureHandler
5. Shared mutable log dicts - Flow via AgentSessionOutput
6. Session config not wired - Fix context_restart_threshold/context_limit propagation
7. Duplicate PromptProvider names - Rename pipeline's to SessionPrompts
8. Locking CLI complexity - Refactor to command dispatch dict

### Out of Scope (Deferred - 2 Low Priority Issues)

- Hook building in pipeline layer
- Config parsing in domain layer

## Assumptions & Constraints

### Implementation Constraints

- **API Breaks**: Allowed freely for internal APIs - no backward-compatibility shims needed
- **No Re-exports**: When moving/renaming modules, update all imports directly
- **Lazy Imports**: SDK imports must remain lazy (local to methods) to preserve bootstrap order

### Testing Constraints

- **Coverage threshold**: 85% (existing gate)
- **New component testing**: Unit tests only for new components
- **Existing tests**: Rely on existing integration/E2E tests as safety net
- **Rollback strategy**: Git revert if issues found

## Prerequisites

- [x] Architecture review completed
- [x] User decisions captured
- [ ] No blocking dependencies

## Implementation Phases

### Phase 1: SDK Boundary + Factory (Issue 1)

**Goal**: Move all Claude SDK usage behind cross-layer protocols; enforce SDK boundary.

**Design**:
- Create `SDKClientFactory` in `src/infra/sdk_adapter.py` with lazy local SDK imports
- Pipeline imports only protocols from `src/core/protocols.py`
- Use `TYPE_CHECKING` imports for SDK types in infra (no runtime dependency)
- Add import-linter rule to prevent SDK imports in pipeline/domain/core

**Files**:
| File | Status | Changes |
|------|--------|---------|
| `src/core/protocols.py` | Modify | Add SDKClientProtocol if needed for cross-layer |
| `src/infra/sdk_adapter.py` | New | SDKClientFactory with lazy SDK imports |
| `src/pipeline/agent_session_runner.py` | Modify | Remove SDK imports, use factory |
| `src/pipeline/run_coordinator.py` | Modify | Inject factory, remove SDK imports |
| `src/orchestration/orchestration_wiring.py` | Modify | Construct and pass factory |
| `pyproject.toml` | Modify | Add `claude_agent_sdk` to forbidden_modules in Contract 2 |

**Tests**:
- Update `tests/test_agent_session_runner.py` to use protocol-based checks
- Verify `tests/test_lazy_imports.py` still passes

**Acceptance Criteria**:
- [ ] No `claude_agent_sdk` imports in `src/pipeline/`, `src/domain/`, `src/core/`
- [ ] Import-linter passes with new rule
- [ ] Existing tests pass

---

### Phase 2: AgentRuntimeBuilder (Issue 2)

**Goal**: Centralize duplicated agent runtime configuration using builder pattern.

**Design**:
- Create `AgentRuntimeBuilder` with fluent API:
  ```python
  AgentRuntimeBuilder(agent_id, repo_path)
      .with_hooks(pre_tool_hooks, stop_hooks)
      .with_env(agent_env)
      .with_mcp(mcp_servers)
      .with_disallowed_tools(tools)
      .build()  # Returns AgentRuntime bundle
  ```
- Bundle contains: options, env, caches, hooks
- Use local SDK imports inside `build()` to preserve lazy-import guarantees

**Files**:
| File | Status | Changes |
|------|--------|---------|
| `src/infra/agent_runtime.py` | New | AgentRuntimeBuilder class |
| `src/pipeline/agent_session_runner.py` | Modify | Replace `_build_agent_env`, `_build_hooks`, `_build_sdk_options` |
| `src/pipeline/run_coordinator.py` | Modify | Replace duplicated setup |
| `tests/test_agent_runtime.py` | New | Unit tests for builder |

**Tests**:
- Validate env composition, hook ordering, disallowed tools wiring
- Test without importing SDK at module level

**Acceptance Criteria**:
- [ ] Single source of truth for agent runtime configuration
- [ ] Both implementer and fixer sessions use builder
- [ ] Unit tests pass for builder

---

### Phase 3: DefaultReviewer Split (Issue 3)

**Goal**: Decompose `DefaultReviewer` into independently testable components.

**Design**:
- `CerberusGateCLI`: subprocess spawn/wait/resolve, binary validation, env merge, timeout handling
- `ReviewOutputParser`: JSON parsing, issue mapping, exit-code mapping, formatting
- `DefaultReviewer`: thin adapter orchestrating CLI + parser, preserves current behavior

**Files**:
| File | Status | Changes |
|------|--------|---------|
| `src/infra/clients/cerberus_gate_cli.py` | New | CLI subprocess management |
| `src/infra/clients/review_output_parser.py` | New | JSON parsing + issue mapping |
| `src/infra/clients/cerberus_review.py` | Modify | Thin adapter using above |
| `tests/test_cerberus_gate_cli.py` | New | CLI command assembly, stale gate flow |
| `tests/test_review_output_parser.py` | New | JSON parsing, exit-code mapping |
| `tests/test_cerberus_review.py` | Modify | Adjust imports to new modules |

**Tests**:
- Use golden JSON fixtures for parser tests
- Test exit-code mapping assertions

**Acceptance Criteria**:
- [ ] `CerberusGateCLI` handles all subprocess spawn/wait/resolve; `DefaultReviewer` has no subprocess calls
- [ ] `ReviewOutputParser` handles all JSON parsing and exit-code mapping; `DefaultReviewer` has no parsing logic
- [ ] Each component independently testable with mocked dependencies
- [ ] Existing Cerberus tests pass

---

### Phase 4: AgentSessionRunner Extraction (Issue 4)

**Goal**: Reduce god class to coordinator by extracting stream processing and context handling.

**Design**:
- `MessageStreamProcessor`: IdleTimeoutStream, IdleTimeoutError, MessageIterationState/Result, stream processing with lint cache updates
- `ContextPressureHandler`: ContextPressureError, checkpoint/restart loop, continuation prompt logic
- `AgentSessionRunner`: becomes coordinator, delegates streaming and context-pressure, keeps lifecycle transitions
- Structure: flat files in `src/pipeline/` (no subdirectories)
- No new protocols (internal components use concrete classes)

**Files**:
| File | Status | Changes |
|------|--------|---------|
| `src/pipeline/message_stream_processor.py` | New | SDK streaming + idle timeout |
| `src/pipeline/context_pressure_handler.py` | New | Checkpoint/restart logic |
| `src/pipeline/agent_session_runner.py` | Modify | Reduce to coordinator (~500 lines target) |
| `tests/test_message_stream_processor.py` | New | Stream parsing, idle retry, lint cache |
| `tests/test_context_pressure_handler.py` | New | Checkpoint fetch, continuation prompts |
| `tests/test_agent_session_runner.py` | Modify | Update to new entry points |

**Tests**:
- Stream parsing with protocol-based message/block checks
- Context pressure trigger at threshold
- Checkpoint fetch timeout handling

**Acceptance Criteria**:
- [ ] `MessageStreamProcessor` handles all SDK stream iteration; no `receive_response()` loops in `AgentSessionRunner`
- [ ] `ContextPressureHandler` handles all checkpoint fetch and restart logic; independently testable without `AgentSessionRunner`
- [ ] `AgentSessionRunner` only coordinates lifecycle transitions and delegates to extracted components
- [ ] Unit tests pass for extracted components

---

### Phase 5: Log Paths via Outputs (Issue 5)

**Goal**: Remove shared mutable `session_log_paths`/`review_log_paths` dictionaries.

**Design**:
- Log paths flow via `AgentSessionOutput` return values
- Add `session_log_path` and `review_log_path` fields to `IssueResult`
- Finalization reads paths from `IssueResult` instead of shared dicts
- Remove dict cleanup helper from orchestrator

**Files**:
| File | Status | Changes |
|------|--------|---------|
| `src/pipeline/issue_result.py` | Modify | Add session_log_path, review_log_path fields |
| `src/orchestration/orchestrator.py` | Modify | Remove dicts, use IssueResult paths |
| `src/pipeline/session_callback_factory.py` | Modify | Remove dict dependencies |
| `src/orchestration/orchestration_wiring.py` | Modify | Remove dicts from WiringDependencies |

**Tests**:
- Update orchestrator tests to read from IssueResult

**Acceptance Criteria**:
- [ ] No shared mutable log path dictionaries
- [ ] Log paths correctly propagate through IssueResult
- [ ] Existing tests pass

---

### Phase 6: Session Config Wiring (Issue 6)

**Goal**: Ensure `context_restart_threshold` and `context_limit` properly propagate.

**Design**:
- Verify wiring: `OrchestratorConfig` → `_build_wiring_dependencies` → `build_session_config` → `AgentSessionConfig`
- Add explicit pass-through or assertion to prevent future regressions

**Files**:
| File | Status | Changes |
|------|--------|---------|
| `src/orchestration/orchestration_wiring.py` | Modify | Ensure explicit wiring |
| `src/orchestration/orchestrator.py` | Modify | Verify wiring deps include fields |
| `src/orchestration/types.py` | Modify (if needed) | Only if fields missing/mis-typed |

**Tests**:
- Add lightweight unit check in `tests/test_orchestrator.py` verifying propagation

**Acceptance Criteria**:
- [ ] Config values propagate from OrchestratorConfig to AgentSessionConfig
- [ ] Unit test verifies propagation

---

### Phase 7: Rename PromptProvider (Issue 7)

**Goal**: Disambiguate naming - rename pipeline's `PromptProvider` → `SessionPrompts`.

**Design**:
- Rename class in `src/pipeline/agent_session_runner.py`
- Keep domain `PromptProvider` unchanged
- Remove aliasing workaround in `orchestration_wiring.py`

**Files**:
| File | Status | Changes |
|------|--------|---------|
| `src/pipeline/agent_session_runner.py` | Modify | Rename class to SessionPrompts |
| `src/orchestration/orchestration_wiring.py` | Modify | Import SessionPrompts, remove alias |
| `tests/test_agent_session_runner.py` | Modify | Update imports |

**Tests**:
- Update affected tests to use `SessionPrompts`

**Acceptance Criteria**:
- [ ] No name collision between pipeline and domain PromptProvider
- [ ] All imports updated

---

### Phase 8: Locking CLI Dispatch (Issue 8)

**Goal**: Simplify `_cli_main` using command dispatch pattern.

**Design**:
- Create `CliContext` dataclass with parsed env
- Create per-command handlers: `_cmd_try`, `_cmd_wait`, `_cmd_check`, `_cmd_holder`, `_cmd_release`, `_cmd_release_all`
- Command dispatch table: `COMMANDS = {'try': _cmd_try, 'wait': _cmd_wait, ...}`
- `_cli_main` validates env/args once and delegates to `COMMANDS[command]`

**Files**:
| File | Status | Changes |
|------|--------|---------|
| `src/infra/tools/locking.py` | Modify | Refactor _cli_main to dispatch pattern |

**Tests**:
- Existing lock script/integration tests should remain valid

**Acceptance Criteria**:
- [ ] `_cli_main` only validates env/args and dispatches to command handlers; no lock operations in main
- [ ] Each command handler (`_cmd_try`, `_cmd_wait`, etc.) performs exactly one lock operation
- [ ] Command dispatch uses `COMMANDS` dict; adding new commands requires only adding handler + dict entry
- [ ] Existing tests pass

---

## File Impact Summary

### New Files (11)

| Path | Purpose |
|------|---------|
| `src/infra/sdk_adapter.py` | SDKClientFactory with lazy SDK imports |
| `src/infra/agent_runtime.py` | AgentRuntimeBuilder |
| `src/infra/clients/cerberus_gate_cli.py` | CLI subprocess management |
| `src/infra/clients/review_output_parser.py` | JSON parsing + issue mapping |
| `src/pipeline/message_stream_processor.py` | SDK streaming + idle timeout |
| `src/pipeline/context_pressure_handler.py` | Checkpoint/restart logic |
| `tests/test_agent_runtime.py` | Unit tests for AgentRuntimeBuilder |
| `tests/test_cerberus_gate_cli.py` | Unit tests for CLI component |
| `tests/test_review_output_parser.py` | Unit tests for parser component |
| `tests/test_message_stream_processor.py` | Unit tests for stream processor |
| `tests/test_context_pressure_handler.py` | Unit tests for context handler |

### Modified Files (12)

| Path | Phases | Changes |
|------|--------|---------|
| `src/core/protocols.py` | 1 | Add SDKClientProtocol if needed |
| `src/pipeline/agent_session_runner.py` | 1,2,4,7 | Remove SDK imports, use builder, extract logic, rename |
| `src/pipeline/run_coordinator.py` | 1,2 | Inject factory, use builder |
| `src/pipeline/session_callback_factory.py` | 5 | Remove dict dependencies |
| `src/pipeline/issue_result.py` | 5 | Add log path fields |
| `src/infra/clients/cerberus_review.py` | 3 | Split into components |
| `src/infra/tools/locking.py` | 8 | Refactor CLI dispatch |
| `src/orchestration/orchestration_wiring.py` | 1,5,6,7 | Wire factory, remove dicts, fix config |
| `src/orchestration/orchestrator.py` | 5,6 | Remove dicts, verify wiring |
| `src/orchestration/types.py` | 6 | Verify config fields if needed |
| `pyproject.toml` | 1 | Add claude_agent_sdk to forbidden_modules |
| `tests/test_agent_session_runner.py` | 1,4,7 | Protocol checks, new entry points |
| `tests/test_cerberus_review.py` | 3 | Point to new modules |

---

## Risks, Edge Cases & Breaking Changes

### Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SDK protocol mismatch | Medium | High | Unit tests using real SDK message objects with strict attribute checks |
| Lazy import regressions | Medium | High | Local imports only; re-run `tests/test_lazy_imports.py` after each phase |
| Behavior drift in hooks/env | Low | Medium | Preserve ordering in builder; test AgentRuntimeBuilder outputs |
| Cerberus behavior change | Low | Medium | Golden JSON fixtures; exit-code mapping tests |
| Test coverage gaps | Low | Medium | Rely on existing integration tests as safety net |

### Breaking Changes

- All internal API changes (acceptable per user decision)
- Import paths for moved/renamed modules
- No external API breaks

### Backwards Compatibility

- Not required - internal APIs can break freely
- No dual-write or versioned code needed

---

## Testing & Validation Strategy

### Unit Tests (New Components Only)

| Component | Test File | Key Tests |
|-----------|-----------|-----------|
| SDKClientFactory | Part of existing tests | Protocol compliance |
| AgentRuntimeBuilder | `tests/test_agent_runtime.py` | Env composition, hook ordering |
| CerberusGateCLI | `tests/test_cerberus_gate_cli.py` | Command assembly, stale gate |
| ReviewOutputParser | `tests/test_review_output_parser.py` | JSON parsing, exit codes |
| MessageStreamProcessor | `tests/test_message_stream_processor.py` | Stream parsing, idle retry |
| ContextPressureHandler | `tests/test_context_pressure_handler.py` | Checkpoint fetch, restart |

### Integration Tests

- Rely on existing integration test suite
- No new integration tests required

### Manual Validation

- Run full orchestrator against real issues after each phase
- Verify import-linter passes

### Acceptance Criteria Coverage

| Review Finding | Phase | Approach |
|----------------|-------|----------|
| SDK imports in pipeline | 1 | Move to infra adapter + import-linter rule |
| Duplicated runtime setup | 2 | AgentRuntimeBuilder with builder pattern |
| DefaultReviewer complexity | 3 | Split into CLI/Parser/Adapter |
| AgentSessionRunner god class | 4 | Extract MessageStreamProcessor, ContextPressureHandler |
| Shared mutable log dicts | 5 | Flow via AgentSessionOutput/IssueResult |
| Config wiring gaps | 6 | Explicit wiring + unit test |
| Duplicate PromptProvider names | 7 | Rename to SessionPrompts |
| Locking CLI complexity | 8 | Command dispatch dict |

---

## Rollback Strategy

- **Per-phase rollback**: Each phase is a complete commit; use `git revert` if issues found
- **Feature flags**: Not required (API breaks allowed)
- **Staged rollout**: Not required for internal refactor

---

## Open Questions

All design questions resolved via user decisions:

| Question | Resolution |
|----------|------------|
| Primary goal | Both testability AND maintainability |
| Scope | 8 issues (defer 2 Low priority) |
| API breaks | Allowed freely |
| AgentRuntimeBuilder pattern | Builder pattern |
| Protocols for extracted components | Only for cross-layer; concrete classes for internal |
| SDK types in infra | TYPE_CHECKING imports |
| Enforcement | Add import-linter rules |

---

## Next Steps

1. Begin Phase 1: SDK Boundary + Factory
2. After each phase: run tests, verify import-linter, commit
3. If regression: git revert and investigate before proceeding
