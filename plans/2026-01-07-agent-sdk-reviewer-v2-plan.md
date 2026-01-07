# Implementation Plan: Agent SDK Code Reviewer

## Context & Goals

- **Spec**: `/home/cyou/mala/plans/2026-01-07-agent-sdk-reviewer-spec.md`
- **Goal**: Implement `AgentSDKReviewer` using true Claude Agent SDK (agent sessions with tools), not Messages API
- **Value**: Reduces installation friction (no Cerberus plugin required) while leveraging Agent SDK's tool capabilities for more thorough code review

## Scope & Non-Goals

### In Scope
- Implement `AgentSDKReviewer` class using `SDKClientFactory` and agent sessions
- Create agent-friendly review prompt (`src/prompts/review_agent.md`) with tool usage guidance
- Update `ValidationConfig` to support `reviewer_type` setting (default: `agent_sdk`, fallback: `cerberus`)
- Update `Factory` to instantiate the appropriate reviewer based on config
- Add unit, integration, and E2E tests for the new reviewer
- Agent has access to git and file reading tools for interactive exploration

### Out of Scope (Non-Goals)
- Changing the behavior of the existing `DefaultReviewer` (Cerberus)
- Adding custom tools beyond standard Agent SDK tools (git, file reading, bash)
- Customizing the Agent SDK client beyond standard configuration
- Real-time streaming of review progress to user

## Assumptions & Constraints

### Assumptions
- `claude_agent_sdk` package is installed and available
- `SDKClientFactory` from `src/infra/sdk_adapter.py` provides the agent creation interface
- Agent SDK provides standard tools (git operations, file reading, bash commands) via the `claude_code` preset with `permission_mode="bypassPermissions"` (see `src/infra/sdk_adapter.py:95-98`)
- Git repository is in a clean state for `git diff` operations
- The JSON schema for review output matches spec (verdict + findings with P0-P3 priorities)

### Implementation Constraints
- **Pattern Matching**: Must use `@dataclass` for the reviewer, similar to `DefaultReviewer`
- **Protocol**: Must strictly adhere to `src/core/protocols.py:CodeReviewer` protocol
- **Dependencies**: Use `SDKClientFactory` for client creation; no direct `claude_agent_sdk` imports outside infra layer
- **Error Handling**: Distinguish between fatal errors (config/setup) and non-fatal errors (API timeouts, parse failures)
- **Tool Access**: Agent must have permission to run git commands and read files in the repository
- **Session Lifecycle**: Properly manage agent session creation, execution, and cleanup

### Testing Constraints
- **Unit Tests**: Must mock the SDK client to avoid network calls and costs
- **E2E Tests**: Mark with `@pytest.mark.e2e` to prevent accidental execution in CI without credentials
- **Coverage**: Ensure handling of empty diffs, malformed JSON responses, API timeouts, agent tool errors

## Prerequisites

- [x] `ANTHROPIC_API_KEY` available in the environment for E2E testing
- [x] `claude_agent_sdk` package installed
- [x] `SDKClientFactory` available in `src/infra/sdk_adapter.py`
- [ ] Agent-friendly review prompt content created (adapt from Cerberus with tool usage guidance)

## High-Level Approach

1. **Infrastructure & Config**: Define configuration schema for `reviewer_type` and create agent-friendly prompt
2. **Agent SDK Integration**: Implement `AgentSDKReviewer` using agent sessions with tools
3. **Factory & Wiring**: Update the system factory to read config and inject the correct reviewer
4. **Verification**: Add comprehensive unit, integration, and E2E tests

## Technical Design

### Architecture

The `AgentSDKReviewer` uses the Agent SDK's agent session pattern instead of simple API calls:

**Data Flow**:
1. `Orchestrator` requests review → calls `AgentSDKReviewer.__call__`
2. `AgentSDKReviewer` creates SDK options via `SDKClientFactory.create_options()`:
   - Uses default `system_prompt={"type": "preset", "preset": "claude_code"}` (provides git/file tools)
   - Sets `permission_mode="bypassPermissions"` (allows tool execution)
   - Review instructions passed in query (step 5), not system prompt
3. `AgentSDKReviewer` creates SDK client via `SDKClientFactory.create(options)`
4. Opens agent session: `async with client:`
5. Sends review request: `await client.query(review_prompt_with_context)`
   - Query contains: review instructions from `review_agent_prompt`, diff range, context
6. Streams agent response: `async for msg in client.receive_response():`
7. Agent uses tools (git diff, file reading) to explore codebase
8. Agent outputs JSON review result
9. `AgentSDKReviewer` parses JSON → `ReviewResult`
10. Session cleanup (handled by async context manager)

**Key Architectural Differences from Messages API**:
- Agent has **interactive tool access** (can run git commands, read files)
- Agent can **explore beyond the provided diff** if needed
- More **complex session lifecycle** (create, query, stream, cleanup)
- **Longer execution time** (agent explores interactively vs single API call)

### Data Model

**Config Schema** (`src/domain/validation/config.py`):
```python
@dataclass
class ValidationConfig:
    # ... existing fields ...
    reviewer_type: Literal["agent_sdk", "cerberus"] = "agent_sdk"
    agent_sdk_review_timeout: int = 600  # Longer than Messages API (agents need time to explore)
    agent_sdk_reviewer_model: str = "sonnet"  # SDK short name format
```

**Prompt Schema** (`src/domain/prompts.py`):
```python
@dataclass
class PromptProvider:
    # ... existing fields ...
    review_agent_prompt: str = ""  # Agent-friendly review instructions
```

**Output**: Reuses existing `ReviewResult` and `ReviewIssue` from `src/infra/clients/review_output_parser.py`

**Protocol Conformance**: `ReviewResult` dataclass fully implements `ReviewResultProtocol` from `src/core/protocols.py:215-235` with all required fields:
- `passed: bool` - Whether the review passed
- `issues: list[ReviewIssue]` - Satisfies `Sequence[ReviewIssueProtocol]`
- `parse_error: str | None` - Parse error message if JSON parsing failed
- `fatal_error: bool` - Whether this is a fatal error (should not retry)
- `review_log_path: Path | None` - Path to review session logs (will be set to SDK session log path)

### API/Interface Design

**`AgentSDKReviewer`** (New Class in `src/infra/clients/agent_sdk_review.py`):

```python
@dataclass
class AgentSDKReviewer:
    """Code reviewer using Claude Agent SDK with tool access.

    Unlike Messages API reviewers, this uses agent sessions where the agent
    can interactively explore the codebase using git and file reading tools.
    """

    repo_path: Path
    review_agent_prompt: str  # Agent-friendly prompt with tool usage guidance
    sdk_client_factory: SDKClientFactoryProtocol
    event_sink: MalaEventSink | None = None
    model: str = "sonnet"  # SDK short name format (sonnet, opus, haiku)
    default_timeout: int = 600

    def overrides_disabled_setting(self) -> bool:
        """Agent SDK reviewer respects disabled setting."""
        return False

    async def __call__(
        self,
        diff_range: str,
        context_file: Path | None = None,
        timeout: int = 600,
        claude_session_id: str | None = None,
        *,
        commit_shas: Sequence[str] | None = None,
    ) -> ReviewResult:
        """Run code review using Agent SDK.

        Creates an agent session with access to git and file reading tools,
        provides review instructions, and collects JSON output.

        Args:
            diff_range: Git diff range (e.g., "HEAD~1..HEAD")
            context_file: Optional file with implementation context
            timeout: Maximum time for agent session (seconds)
            claude_session_id: Optional session ID for telemetry
            commit_shas: Specific commit SHAs being reviewed

        Returns:
            ReviewResult with verdict and findings
        """
        # Implementation steps:
        # 1. Check empty diff → short-circuit with PASS
        # 2. Load context file (if provided)
        # 3. Create SDK options (with git/file tools enabled)
        # 4. Create SDK client
        # 5. Open agent session (async with)
        # 6. Send review query with diff range and context
        # 7. Stream agent response messages
        # 8. Extract JSON from final message
        # 9. Parse JSON → ReviewResult (with review_log_path set to SDK session log)
        # 10. Emit telemetry: on_review_warning for non-fatal errors (matches DefaultReviewer pattern)
        #     Note: issue_id parameter is optional and not available in CodeReviewer.__call__ signature
        ...

    async def _check_diff_empty(self, diff_range: str) -> bool:
        """Check if diff range is empty using git diff --stat."""
        ...

    async def _load_context(self, context_file: Path | None) -> str:
        """Load context file asynchronously."""
        ...

    def _create_review_query(
        self,
        diff_range: str,
        context: str,
        commit_shas: Sequence[str] | None
    ) -> str:
        """Construct review query for agent.

        Combines:
        - Review instructions (from review_agent_prompt)
        - Diff range to review
        - Implementation context
        - Tool usage guidance (how to use git/file tools)
        """
        ...

    async def _run_agent_session(
        self,
        query: str,
        timeout: int,
        session_id: str | None,
    ) -> str:
        """Run agent session and collect final output.

        Creates SDK client, opens session, sends query, streams responses,
        and extracts final JSON output.
        """
        ...

    def _parse_response(self, response_text: str) -> ReviewResult:
        """Parse JSON response into ReviewResult.

        Handles:
        - JSON extraction from markdown code blocks
        - Fallback: find first { and last }
        - Priority conversion (string "P1" → int 1)
        - Verdict normalization (PASS/FAIL/NEEDS_WORK)
        """
        ...
```

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/domain/validation/config.py` | Modify | Add `reviewer_type`, `agent_sdk_review_timeout`, `agent_sdk_reviewer_model` fields |
| `src/domain/prompts.py` | Modify | Add `review_agent_prompt` field to `PromptProvider` |
| `src/prompts/review_agent.md` | **New** | Agent-friendly review system prompt with tool usage guidance |
| `src/infra/clients/agent_sdk_review.py` | **New** | Implementation of `AgentSDKReviewer` using Agent SDK |
| `src/orchestration/factory.py` | Modify | Add logic to switch between `AgentSDKReviewer` and `DefaultReviewer` based on config |
| `tests/unit/infra/test_agent_sdk_review.py` | **New** | Unit tests for `AgentSDKReviewer` |
| `tests/integration/test_agent_sdk_integration.py` | **New** | Integration tests for factory wiring and ReviewRunner integration |
| `tests/e2e/test_agent_sdk_e2e.py` | **New** | E2E test with real Agent SDK (marked `@pytest.mark.e2e`) |
| `docs/project-config.md` | Modify | Document new `reviewer_type` configuration option |

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes

**Empty Diff**:
- *Handling*: Check diff size before creating agent session. If empty, return "Passed" result immediately.
- *Test*: Unit test with `git diff --stat` returning empty output

**Malformed JSON**:
- *Handling*: Catch `JSONDecodeError`. Return `ReviewResult` with `success=False` and error details.
- *Test*: Unit test with agent returning non-JSON text

**Agent Tool Errors**:
- *Scenario*: Agent tries to run `git diff` but encounters permission error or git is not available
- *Handling*: Agent SDK should handle tool errors gracefully. Monitor for tool error messages in stream.
- *Test*: Integration test with git unavailable (mock git command failure)

**Context Length Exceeded**:
- *Scenario*: Large diff + exploration causes context overflow
- *Handling*: Agent SDK should handle context pressure. If it fails, catch exception and return error result.
- *Test*: Unit test with mocked context pressure exception

**Timeout**:
- *Scenario*: Agent session exceeds timeout (exploring too much, slow API)
- *Handling*: Use `asyncio.wait_for()` to enforce timeout. On `TimeoutError`, return result with timeout message.
- *Test*: Unit test with mocked slow agent response

**Agent Doesn't Return JSON**:
- *Scenario*: Agent explores but doesn't output required JSON format
- *Handling*: Fallback JSON extraction (find first `{` and last `}`). If still fails, return error result.
- *Test*: Unit test with agent returning plain text explanation instead of JSON

**Session Creation Failure**:
- *Scenario*: Missing API key, network error, SDK initialization failure
- *Handling*: Catch exception during SDK client creation. Return fatal error result.
- *Test*: Unit test with mocked SDK client factory raising exception

### Breaking Changes & Compatibility

**Default Change**:
- *Change*: Default `reviewer_type` changes from `cerberus` to `agent_sdk`
- *Impact*: Users without Cerberus plugin installed will now get working reviews (good!)
- *Mitigation*: Users who prefer Cerberus can set `reviewer_type: cerberus` in `mala.yaml`
- *Compatibility*: The `CodeReviewer` protocol ensures the rest of the system is unaffected

**Timeout Change**:
- *Change*: Agent SDK reviewer has longer default timeout (600s vs Cerberus 1200s, Messages API 300s)
- *Rationale*: Agents need time for interactive exploration
- *Mitigation*: Configurable via `agent_sdk_review_timeout`

**Performance Impact**:
- *Change*: Agent SDK reviews may be slower than Messages API (interactive exploration)
- *Benefit*: More thorough reviews (agent can explore beyond provided diff)
- *Mitigation*: Users who want fast reviews can use Messages API (future implementation) or Cerberus

## Testing & Validation Strategy

### Unit Tests (`tests/unit/infra/test_agent_sdk_review.py`)

**Test Cases** (12 tests):
1. **`test_empty_diff_skips_agent_session`**: Verify empty diff short-circuits without creating agent session
2. **`test_empty_commit_sha_skips_agent_session`**: Verify empty commit_shas short-circuits without creating agent session
3. **`test_successful_review_pass`**: Mock agent returning valid JSON with PASS verdict
4. **`test_successful_review_fail`**: Mock agent returning valid JSON with FAIL verdict and findings
5. **`test_successful_review_needs_work`**: Mock agent returning NEEDS_WORK with mixed priority findings
6. **`test_malformed_json_response`**: Mock agent returning non-JSON text, verify graceful error handling
7. **`test_timeout_handling`**: Mock slow agent response exceeding timeout, verify TimeoutError handling
8. **`test_sdk_client_creation_failure`**: Mock SDK factory raising exception, verify fatal error result
9. **`test_context_file_loading`**: Verify context file is loaded asynchronously and passed to agent
10. **`test_priority_conversion`**: Verify both string ("P1") and int (1) priority formats are handled
11. **`test_telemetry_warning_on_error`**: Verify event sink receives on_review_warning for non-fatal errors
12. **`test_overrides_disabled_setting_returns_false`**: Verify overrides_disabled_setting() returns False

**Mocking Strategy**:
- Mock `SDKClientFactory` to return fake SDK client
- Mock SDK client's `query()` and `receive_response()` methods
- Mock `git diff --stat` subprocess calls
- Mock async file I/O for context loading

### Integration Tests (`tests/integration/test_agent_sdk_integration.py`)

**Test Cases** (4 tests):
1. **`test_factory_creates_agent_sdk_reviewer_by_default`**: Verify factory creates `AgentSDKReviewer` when no config specified
2. **`test_factory_creates_cerberus_reviewer_when_configured`**: Verify factory creates `DefaultReviewer` when `reviewer_type: cerberus`
3. **`test_factory_creates_agent_sdk_reviewer_when_configured`**: Verify factory creates `AgentSDKReviewer` when `reviewer_type: agent_sdk`
4. **`test_review_runner_integration`**: Verify `ReviewRunner` → `AgentSDKReviewer` → `ReviewResult` end-to-end flow with mocked SDK

**Note**: Integration tests require factory wiring (T005) to be complete first. Tests should traverse the full factory → reviewer creation path.

**Setup**:
- Use real factory with test configuration
- Mock SDK client to avoid network calls
- Verify wiring between components

### E2E Tests (`tests/e2e/test_agent_sdk_e2e.py`)

**Test Cases**:
1. **`test_real_agent_review_flow`**:
   - Requires `ANTHROPIC_API_KEY`
   - Create test git repository with simple code change
   - Run `AgentSDKReviewer` with real Agent SDK
   - Verify valid `ReviewResult` structure returned
   - Verify agent used tools (check for git diff tool usage in logs)

**Constraints**:
- Mark with `@pytest.mark.e2e`
- Skip if `ANTHROPIC_API_KEY` not set
- Use small, cheap test cases to minimize API costs
- Clean up test repositories after execution

### Manual Verification

**Scenarios**:
1. **Fresh install without Cerberus**: Verify mala works out-of-box with Agent SDK reviewer
2. **Large diff review**: Verify agent can handle multi-file changes with deep exploration
3. **Edge case: binary files in diff**: Verify agent handles git diff with binary files gracefully
4. **Config override**: Verify switching to Cerberus via `reviewer_type: cerberus` works

**Environments**:
- Clean Python environment (fresh venv)
- Repository with various diff sizes (small, medium, large)
- Different git configurations (shallow clone, deep history)

### Acceptance Criteria Coverage

| Spec AC | Covered By |
|---------|------------|
| R1: Protocol Compatibility | `AgentSDKReviewer.__call__` signature matches `CodeReviewer` protocol, Unit tests verify empty diff handling |
| R2: Configurable (Default SDK) | `ValidationConfig.reviewer_type` defaults to `agent_sdk`, Factory integration tests verify switching |
| R3: Prompt Parity | `src/prompts/review_agent.md` adapted from Cerberus with tool usage guidance |
| R4: JSON Output Normalization | `_parse_response` method handles priority conversion, verdict normalization; Unit tests verify parsing |
| R5: Timeout Handling | `asyncio.wait_for()` enforces timeout, Unit tests verify timeout exceptions |
| R6: In-Process Execution | Agent SDK runs in-process (no subprocess calls except tools), Architecture section confirms |
| R7: Empty Diff Handling | `_check_diff_empty` short-circuits before agent session creation, Unit test verifies |

## Open Questions

None. All implementation details are resolved based on existing patterns and Agent SDK documentation.

## Implementation Order

**IMPORTANT**: Task ordering ensures factory wiring happens before integration tests run.

1. **Phase 1: Core Infrastructure** (T001-T002, parallel)
   - T001: Create `src/prompts/review_agent.md` (agent-friendly prompt with tool guidance)
   - T002: Add config fields to `ValidationConfig` and `PromptProvider`

2. **Phase 2: Skeleton + Factory Wiring** (T003)
   - T003: Create `src/infra/clients/agent_sdk_review.py` with skeleton class
   - T003: Wire skeleton into `src/orchestration/factory.py` (so factory knows about AgentSDKReviewer)
   - Dependencies: T001, T002

3. **Phase 3: Core Implementation** (T004)
   - T004: Implement `AgentSDKReviewer` core logic with 12 unit tests:
     - Empty diff check (diff_range and commit_shas)
     - Context loading
     - SDK client creation
     - Agent session lifecycle
     - JSON parsing
     - Telemetry (on_review_warning)
     - overrides_disabled_setting
   - Dependencies: T003

4. **Phase 4: Integration Tests** (T005)
   - T005: Add integration tests for factory wiring and ReviewRunner
   - T005: Verify all 4 integration test cases pass
   - Dependencies: T004 (implementation must be complete for tests to pass)

5. **Phase 5: E2E & Documentation** (T006-T007, parallel)
   - T006: Add E2E test with real Agent SDK (marked `@pytest.mark.e2e`)
   - T007: Update `docs/project-config.md` with new config options
   - Dependencies: T005

## Next Steps

Run `/cerberus:create-tasks --beads` to generate Beads issues with dependencies for parallel execution.
