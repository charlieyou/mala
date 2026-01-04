# Implementation Plan: Architecture Refactoring

## Context & Goals
- **Source**: Architecture review at `~/.claude/projects/-home-cyou-mala/cerberus/.../latest.md`
- [TBD: Confirm primary goal - testability vs maintainability vs both]
- [TBD: Success metrics for the refactor]
- [TBD: Timeline expectations or phasing preferences]

## Scope & Non-Goals

### In Scope
- 10 architecture issues from review (3 High, 5 Medium, 2 Low)
- SDK boundary enforcement in pipeline layer
- Agent runtime unification (implementer + fixer)
- `DefaultReviewer` decomposition
- `AgentSessionRunner` god class decomposition
- Shared mutable state elimination
- Config wiring improvements
- Naming clarifications

### Out of Scope (Non-Goals)
- [TBD: Any issues to defer or exclude?]
- [TBD: Performance optimizations beyond complexity reduction?]
- [TBD: New features or capability additions?]

## Assumptions & Constraints

### Implementation Constraints
- [TBD: Can we break backward compatibility for internal APIs?]
- [TBD: Any modules that must remain untouched?]
- [TBD: Feature flag requirements for rollout?]

### Testing Constraints
- Coverage threshold: 85% (existing gate)
- [TBD: Required test types for new components - unit only or integration too?]
- [TBD: Are there integration test patterns to follow?]

## Prerequisites
- [ ] [TBD: Any approvals needed?]
- [ ] [TBD: Dependencies on other work?]
- [ ] [TBD: Infrastructure or tooling updates?]

## High-Level Approach

[TBD: Phased approach or big-bang? Suggested phases:]

### Phase 1: Foundation - Agent Runtime Builder
[TBD: Details on extracting runtime configuration to infra]

### Phase 2: SDK Boundary Enforcement
[TBD: How to move SDK imports from pipeline to infra]

### Phase 3: Cerberus Adapter Split
[TBD: Decomposing DefaultReviewer into CLI/Parser/Adapter]

### Phase 4: AgentSessionRunner Decomposition
[TBD: Extracting MessageStreamProcessor, ContextPressureHandler, etc.]

### Phase 5: Cleanup
[TBD: Shared state elimination, config wiring, naming fixes]

## Technical Design

### Architecture

#### New Component: `AgentRuntimeBuilder` (infra layer)
- Location: `src/infra/agent_runtime.py` (New)
- Purpose: [TBD: Centralize env + hooks + MCP + disallowed tools]
- Interface: [TBD: Builder pattern vs factory vs config object?]

#### New Component: SDK Adapter (infra layer)
- [TBD: `SDKClientFactory` vs direct protocol injection?]
- [TBD: How to handle type hints for SDK types?]

#### Cerberus Decomposition
- `CerberusGateCLI` (New): [TBD: Subprocess management only?]
- `ReviewOutputParser` (New): [TBD: JSON parsing + issue mapping?]
- `DefaultReviewer` (refactored): [TBD: Thin adapter role?]

#### AgentSessionRunner Decomposition
- `MessageStreamProcessor` (New): [TBD: SDK streaming + idle timeout?]
- `ContextPressureHandler` (New): [TBD: Checkpoint/restart logic?]
- `SessionRetryPolicy` (New): [TBD: Backoff decisions?]
- `AgentSessionRunner` (refactored): [TBD: Coordinator only?]

### Data Model
- [TBD: Any new data classes needed?]
- [TBD: Changes to existing data classes?]

### API/Interface Design

#### New Protocols
- [TBD: Do we need new protocols for extracted components?]
- [TBD: Should `AgentRuntimeBuilder` have a protocol?]

#### Modified Protocols
- [TBD: Any changes to existing protocols in core/protocols.py?]

### File Impact Summary

**Existing files to modify:**
- `src/pipeline/agent_session_runner.py` — Exists (extract logic, reduce to coordinator)
- `src/pipeline/run_coordinator.py` — Exists (use AgentRuntimeBuilder)
- `src/infra/clients/cerberus_review.py` — Exists (split into 3 components)
- `src/orchestration/orchestration_wiring.py` — Exists (fix config propagation)
- `src/orchestration/types.py` — Exists (verify config fields)
- `src/domain/prompts.py` — Exists (keep PromptProvider name)
- `src/pipeline/session_callback_factory.py` — Exists (remove shared mutable dicts)
- `src/infra/tools/locking.py` — Exists (refactor CLI to command pattern)

**New files to create:**
- `src/infra/agent_runtime.py` — New (AgentRuntimeBuilder)
- `src/infra/clients/cerberus_gate_cli.py` — New (CLI subprocess management)
- `src/infra/clients/review_output_parser.py` — New (JSON parsing)
- `src/pipeline/message_stream_processor.py` — New (SDK streaming)
- `src/pipeline/context_pressure_handler.py` — New (checkpoint logic)
- `tests/infra/test_agent_runtime.py` — New
- `tests/infra/clients/test_cerberus_gate_cli.py` — New
- `tests/infra/clients/test_review_output_parser.py` — New
- `tests/pipeline/test_message_stream_processor.py` — New

## Risks, Edge Cases & Breaking Changes

### Risks
- [TBD: Risk of regressions in critical paths?]
- [TBD: Risk of test coverage gaps during transition?]
- [TBD: Concurrent session behavior changes?]

### Breaking Changes
- [TBD: Any internal API breaks that affect tests?]
- [TBD: Import path changes?]

### Backwards Compatibility
- [TBD: Do we need dual-write or versioned code during transition?]

## Testing & Validation Strategy

### Unit Tests
- [TBD: Required for all new components?]
- [TBD: Mock SDK for MessageStreamProcessor tests?]

### Integration Tests
- [TBD: Full session flow tests?]
- [TBD: Cerberus integration tests?]

### Manual Validation
- [TBD: Run against real issues?]
- [TBD: Specific scenarios to validate?]

### Acceptance Criteria Coverage
| Review Finding | Plan Approach |
|----------------|---------------|
| SDK imports in pipeline | [TBD] |
| Duplicated runtime setup | [TBD] |
| DefaultReviewer complexity | [TBD] |
| AgentSessionRunner god class | [TBD] |
| Shared mutable log dicts | [TBD] |
| Config wiring gaps | [TBD] |
| Duplicate PromptProvider names | [TBD] |
| Locking CLI complexity | [TBD] |
| Hook building in pipeline | [TBD] |
| Config parsing in domain | [TBD] |

## Rollback Strategy
- [TBD: Feature flags for new code paths?]
- [TBD: Git-based rollback sufficient?]
- [TBD: Staged rollout to catch issues?]

## Open Questions
1. Should we enforce SDK boundary with import-linter rules now or defer?
2. Is `AgentRuntimeBuilder` the right abstraction or should it be `AgentRuntimeConfig`?
3. Should extracted components get their own protocols or use concrete classes?
4. How to handle TYPE_CHECKING imports for SDK types in infra layer?
5. Should Low priority issues (hook building, config parsing) be included or deferred?

## Next Steps
After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
