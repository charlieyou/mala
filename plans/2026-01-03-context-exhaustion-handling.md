# Implementation Plan: Context Exhaustion Handling

## Context & Goals
- **Spec**: `docs/2026-01-03-context-exhaustion-skeleton.md`
- Handle the 200K token context limit gracefully in mala agent orchestrator
- Treat context exhaustion as a **session boundary**, not a process boundary
- Enable seamless continuation by immediately starting a new SDK session
- Target outcomes: zero hard crashes at context limit, seamless continuation, no overhead

## Scope & Non-Goals

### In Scope
- Layer 1: Context pressure detection via SDK token usage monitoring (90% threshold triggers restart)
- Layer 2: Agent-generated checkpoint summary via injected prompt
- Layer 3: Immediate new session with checkpoint as initial context
- Integration with existing lifecycle state machine

### Out of Scope (Non-Goals)
- Checkpoint persistence to filesystem
- Continuation issue creation (same process continues immediately)
- Safety checks before continuation (repo can stay dirty)
- Git commits for partial work
- Lock release/reacquisition (same process keeps locks)
- Continuation depth limits (trust the agent)
- Increasing context window size (model limitation)
- Modifying Agent SDK internals

## Assumptions & Constraints

### Assumptions
- The Agent SDK `ResultMessage` exposes token usage data via `message.usage` field
- Agents will comply with injected prompts to output structured `<checkpoint>` blocks
- SDK allows starting a new session from the same process while keeping environment state

### Implementation Constraints
- Extend existing `LifecycleContext` dataclass for token tracking
- Must integrate with existing `LifecycleState`/`Effect` enums without breaking changes
- Keep existing file locks across session boundaries
- No external state persistence required

### Testing Constraints
- Unit tests for lifecycle state transitions with context pressure
- Unit tests for checkpoint parsing
- Integration tests with mocked SDK for session restart flow
- Must run ruff + ty checks before merge (enforced by quality gate)

## Prerequisites
- [x] Verify SDK provides token usage in `ResultMessage.usage` field
- [x] Verify SDK supports starting new session from same process
- [ ] Ensure all existing tests pass before starting implementation

## High-Level Approach

The key insight: **context exhaustion is a session boundary, not a process boundary**. Instead of creating a continuation issue and waiting for it to be picked up, we immediately start a new SDK session with the checkpoint context.

```
Session 1: Working on task...
  │
  ├─ Hit 90% context threshold
  ├─ Inject checkpoint prompt
  ├─ Agent outputs <checkpoint> block
  ├─ Parse checkpoint (in-memory only)
  │
  └─ Immediately start Session 2
       │
       ├─ Initial prompt = checkpoint context + "continue from remaining tasks"
       ├─ Same process, same locks, same repo state
       └─ Continue working...
```

Benefits over continuation-issue approach:
- **No latency**: No waiting for orchestrator to pick up continuation issue
- **No lock churn**: Same process keeps locks, no release/reacquire
- **No persistence overhead**: No checkpoint files, no issue creation
- **No safety checks needed**: Repo stays in whatever state, next session continues
- **Simpler implementation**: ~50% fewer tasks than original plan

## Technical Design

### Architecture

```
AgentSessionRunner.run_session() [modified - outer restart loop]
  │
  └── while True:
        │
        ├── Create fresh LifecycleContext (with context_usage, continuation_count preserved)
        ├── Create fresh ImplementerLifecycle instance
        │
        ├── _run_lifecycle_loop() [existing - inner loop]
        │     ├── lifecycle.start()
        │     ├── _run_message_iteration()
        │     │     └── _process_message_stream()
        │     │           ├── Extract token usage from ResultMessage.usage
        │     │           ├── Update lifecycle_ctx.context_usage
        │     │           └── On 90% threshold: raise ContextPressureError
        │     └── Handle gate/review as normal
        │
        ├── Catch ContextPressureError:
        │     ├── Send CHECKPOINT_REQUEST_PROMPT via new SDK query
        │     ├── Parse checkpoint from response
        │     ├── Build continuation prompt
        │     ├── Clear session_id (forces new SDK session)
        │     ├── Increment continuation_count
        │     └── Continue loop with new prompt
        │
        └── Normal completion: return AgentSessionOutput
```

**Key design decisions:**
1. **Fresh lifecycle per restart**: Each iteration creates new `LifecycleContext` and `ImplementerLifecycle` to satisfy `start()` requiring INITIAL state. Only `continuation_count` is preserved across restarts.
2. **Clear session_id on restart**: Setting `session_id = None` forces SDK to create a fresh session with empty context.
3. **Exception-based signaling**: Use `ContextPressureError` exception to cleanly exit `_process_message_stream()` without modifying `MessageIterationResult` type.
4. **Checkpoint query after stream ends**: Don't inject mid-stream. Let current stream complete/abort, then send checkpoint prompt as a new query to the existing session before starting fresh.

### Data Model

```python
@dataclass
class ContextUsage:
    """Token usage tracking for context pressure detection.

    Note: SDK provides CUMULATIVE input_tokens (total context window usage),
    not incremental per-message tokens. This is verified in prerequisites.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    def pressure_ratio(self, limit: int = 200_000) -> float:
        """Return total tokens as fraction of limit (0.0 to 1.0+).

        Note: cache_read_tokens are already included in input_tokens
        per the Anthropic API, so we only sum input + output.
        """
        if limit <= 0:
            return 0.0
        return (self.input_tokens + self.output_tokens) / limit


class ContextPressureError(Exception):
    """Raised when context usage exceeds restart threshold.

    Caught by run_session() to trigger checkpoint and restart flow.
    Contains the session_id needed to send checkpoint prompt.
    """
    def __init__(self, session_id: str, usage: ContextUsage):
        self.session_id = session_id
        self.usage = usage
        super().__init__(f"Context pressure: {usage.pressure_ratio():.1%}")
```

**Checkpoint is just a string** - the extracted `<checkpoint>` block content (or full response as fallback). No structured parsing needed; the continuation prompt template handles it.

### API/Interface Design

**No new lifecycle effects needed** - using exception-based signaling (`ContextPressureError`) instead of adding to `Effect` enum. This avoids modifying `MessageIterationResult` and keeps the existing lifecycle state machine unchanged.

New config fields in `src/orchestration/types.py`:

```python
@dataclass
class OrchestratorConfig:
    # ... existing fields ...
    context_restart_threshold: float = 0.90    # 180K tokens - restart session
    context_limit: int = 200_000               # Model context window
```

New field on `LifecycleContext`:

```python
@dataclass
class LifecycleContext:
    # ... existing fields ...
    context_usage: ContextUsage = field(default_factory=ContextUsage)
```

New exception in `src/pipeline/agent_session_runner.py`:

```python
class ContextPressureError(Exception):
    """Raised when context usage exceeds restart threshold."""
    session_id: str
    usage: ContextUsage
```

**Session-level state** (not on LifecycleContext, managed by run_session):

```python
# Tracked across restart iterations in run_session()
continuation_count: int = 0  # Incremented on each restart
current_prompt: str          # Updated with continuation prompt on restart
```

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/domain/lifecycle.py` | Modify | Add ContextUsage dataclass, add context_usage field to LifecycleContext |
| `src/orchestration/types.py` | Modify | Add context_restart_threshold and context_limit to OrchestratorConfig |
| `src/pipeline/agent_session_runner.py` | Modify | Add ContextPressureError, token tracking, checkpoint handling, session restart loop in run_session() |
| `src/prompts/checkpoint_request.md` | **New** | Prompt asking agent to output checkpoint summary |
| `src/prompts/continuation.md` | **New** | Continuation prompt template with checkpoint context |
| `tests/domain/test_lifecycle.py` | Modify | Add ContextUsage tests |
| `tests/pipeline/test_agent_session_runner.py` | Modify | Add session restart tests |

## Detailed Plan

### Task 1: Add ContextUsage Dataclass and Config

- **Goal**: Define data structures for token tracking and thresholds
- **Depends on**: None
- **Changes**:
  - `src/domain/lifecycle.py`:
    - Add `ContextUsage` dataclass with `input_tokens`, `output_tokens`, `cache_read_tokens`
    - Add `pressure_ratio(limit: int) -> float` method
    - Add `context_usage: ContextUsage` field to `LifecycleContext`
  - `src/orchestration/types.py`:
    - Add `context_restart_threshold: float = 0.90` to `OrchestratorConfig`
    - Add `context_limit: int = 200_000` to `OrchestratorConfig`
- **Verification**:
  - Unit test `ContextUsage.pressure_ratio()` at 0%, 50%, 100%
  - Unit test config defaults
- **Rollback**: Revert changes to lifecycle.py, types.py

### Task 2: Token Usage Extraction and Context Pressure Detection

- **Goal**: Extract token usage from SDK and raise exception on threshold
- **Depends on**: Task 1
- **Changes**:
  - `src/pipeline/agent_session_runner.py`:
    - Add `ContextPressureError(Exception)` with `session_id` and `usage` fields
    - In `_process_message_stream()`, after receiving `ResultMessage`:
      ```python
      if hasattr(message, 'usage') and message.usage is not None:
          usage = message.usage
          lifecycle_ctx.context_usage.input_tokens = getattr(usage, 'input_tokens', 0)
          lifecycle_ctx.context_usage.output_tokens = getattr(usage, 'output_tokens', 0)
          lifecycle_ctx.context_usage.cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0)

          ratio = lifecycle_ctx.context_usage.pressure_ratio(config.context_limit)
          if ratio >= config.context_restart_threshold:
              raise ContextPressureError(state.session_id, lifecycle_ctx.context_usage)
      else:
          # Graceful fallback: log warning and disable context tracking for this session
          logger.warning(f"Session {issue_id}: SDK did not provide token usage, context tracking disabled")
          lifecycle_ctx.context_usage.input_tokens = -1  # Sentinel: tracking disabled
      ```
- **Verification**:
  - Unit test with mocked ResultMessage - verify extraction
  - Unit test without usage field - verify warning logged, no crash
  - Unit test at 90% threshold - verify ContextPressureError raised
- **Rollback**: Revert changes to agent_session_runner.py

### Task 3: Checkpoint Prompts

- **Goal**: Define prompts for checkpoint generation and continuation as markdown files
- **Depends on**: None
- **Changes**:
  - **New**: `src/prompts/checkpoint_request.md`:
    ```markdown
    CONTEXT LIMIT APPROACHING - CHECKPOINT REQUIRED

    Output a checkpoint summary now:

    <checkpoint>
    ## Goal
    [1-2 sentence summary of the original task]

    ## Completed Work
    [List each file modified and what changed]

    ## Remaining Tasks
    [Ordered list of what still needs to be done]

    ## Do Not Redo
    [Completed items that should not be repeated]

    ## Key Decisions
    [Architectural choices or constraints for continuation]
    </checkpoint>

    After outputting this checkpoint, STOP.
    ```
  - **New**: `src/prompts/continuation.md`:
    ```markdown
    CONTINUING FROM PREVIOUS SESSION

    The previous session hit the context limit. Here is the state:

    {checkpoint}

    Continue from the remaining tasks. Do not repeat completed work.
    ```
  - `src/domain/prompts.py`:
    - Add `load_prompt(name: str) -> str` to load from `src/prompts/{name}.md`
    - Add `build_continuation_prompt(checkpoint_text: str) -> str` that loads `continuation.md` and formats with checkpoint
- **Verification**:
  - Unit test: load_prompt loads markdown files correctly
  - Unit test: build_continuation_prompt formats checkpoint into template
- **Rollback**: Delete new .md files, revert prompts.py

### Task 4: Checkpoint Parsing

- **Goal**: Extract checkpoint block from agent response
- **Depends on**: Task 3
- **Changes**:
  - `src/domain/prompts.py`:
    - Add `extract_checkpoint(text: str) -> str`:
      - Strip markdown code block wrappers (```xml, ```markdown, ``` etc.)
      - Extract content between `<checkpoint>` and `</checkpoint>` tags
      - If no tags found, return the full response text as fallback
      - Return the extracted/fallback text (no structured parsing needed)
- **Verification**:
  - Unit test: extract well-formed checkpoint block
  - Unit test: extract checkpoint wrapped in ```xml code block
  - Unit test: fallback to full text when no tags found
- **Rollback**: Revert changes to prompts.py

### Task 5: Session Restart Loop in run_session()

- **Goal**: Implement restart loop with fresh lifecycle per iteration
- **Depends on**: Tasks 1, 2, 3, 4
- **Changes**:
  - `src/pipeline/agent_session_runner.py`:
    - Modify `run_session()` to wrap execution in restart loop:
      ```python
      async def run_session(self, input: AgentSessionInput, ...) -> AgentSessionOutput:
          current_prompt = input.prompt
          continuation_count = 0

          while True:
              # Create fresh lifecycle state for each iteration
              # (lifecycle.start() requires INITIAL state)
              lifecycle_ctx = LifecycleContext()
              lifecycle = ImplementerLifecycle(self.lifecycle_config)

              try:
                  # Run existing lifecycle loop (calls lifecycle.start() internally)
                  output = await self._run_lifecycle_loop(
                      current_prompt, lifecycle, lifecycle_ctx, ...
                  )
                  return output  # Normal completion

              except ContextPressureError as e:
                  # Get checkpoint from agent before starting new session
                  checkpoint_text = await self._get_checkpoint_from_agent(
                      e.session_id, input.issue_id
                  )

                  continuation_count += 1
                  logger.info(
                      f"Session {input.issue_id}: context restart #{continuation_count} "
                      f"at {e.usage.pressure_ratio():.1%}"
                  )

                  # Build continuation prompt with checkpoint (no original task needed)
                  current_prompt = build_continuation_prompt(checkpoint_text)

                  # Next iteration creates new lifecycle_ctx with session_id=None
                  # This forces SDK to start a fresh session
                  continue
      ```
    - Add `_get_checkpoint_from_agent()` method:
      ```python
      async def _get_checkpoint_from_agent(
          self, session_id: str, issue_id: str
      ) -> str:
          """Send checkpoint prompt to current session and extract response."""
          checkpoint_prompt = load_prompt("checkpoint_request")
          # Use same SDK client factory as main session
          client = self.sdk_client_factory.create(self.session_cfg.options)
          async with client:
              await client.query(checkpoint_prompt, session_id=session_id)
              response_text = ""
              async for message in client.receive_response():
                  if isinstance(message, AssistantMessage):
                      for block in message.content:
                          if isinstance(block, TextBlock):
                              response_text += block.text
          return extract_checkpoint(response_text)
      ```
- **Verification**:
  - Integration test: simulate 90% threshold, verify restart occurs
  - Integration test: verify fresh lifecycle created each iteration
  - Integration test: verify session_id cleared (new SDK session)
  - Integration test: verify continuation prompt contains checkpoint
  - Integration test: verify continuation_count increments
- **Rollback**: Revert changes to agent_session_runner.py

### Task 6: Integration Tests and Cleanup

- **Goal**: Verify end-to-end flow, clean up code
- **Depends on**: Tasks 1-5
- **Changes**:
  - `tests/pipeline/test_agent_session_runner.py`:
    - Test: session hits 90% → ContextPressureError → checkpoint → restart with continuation
    - Test: multiple restarts in sequence (continuation_count = 1, 2, 3...)
    - Test: SDK missing usage field → warning logged, session continues without tracking
    - Test: checkpoint without tags → full response used as fallback
    - Test: fresh lifecycle created each restart (no "Cannot start from state X" error)
  - Run `uvx ruff format .` and `uvx ruff check . --fix`
  - Verify `uvx ty check` passes
  - Verify full test suite passes
- **Verification**:
  - All tests pass
  - No linting errors
- **Rollback**: Delete test additions (no production impact)

## Risks, Edge Cases & Breaking Changes

### Risks
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SDK doesn't expose token usage | Low | Medium | Graceful fallback: log warning, disable tracking, session continues |
| Agent ignores checkpoint prompt | Medium | Medium | Fallback to raw text in continuation prompt |
| Checkpoint parsing fails | Medium | Low | Use CONTINUATION_FALLBACK_TEMPLATE with raw_text |
| Infinite restart loop | Low | Medium | Logged continuation_count; human/timeout can kill |
| Token spike past threshold | Low | Low | 90% threshold leaves 20K buffer for checkpoint prompt |
| Checkpoint prompt rejected (context full) | Low | Medium | Catch API error, restart with minimal "continue previous work" prompt |
| Lifecycle start() crash on restart | N/A | N/A | **Resolved**: Fresh LifecycleContext + ImplementerLifecycle per iteration |

### Edge Cases
- **SDK missing usage data**: Log warning, set sentinel (-1), session continues without context tracking
- **Malformed checkpoint block**: Parse what's possible, store raw_text, use CONTINUATION_FALLBACK_TEMPLATE
- **Agent doesn't output checkpoint tags**: Use entire response as raw_text fallback
- **Checkpoint prompt exceeds remaining context**: Catch `PromptTooLong` error, restart with minimal continuation prompt
- **Process crash during restart**: Same as today - work lost, no worse than before
- **Token count is incremental not cumulative**: Prerequisite verification confirms SDK provides cumulative; if wrong, thresholds will trigger too early (safe failure mode)

### Breaking Changes
- **None**: All additions are backward compatible
- New config fields have safe defaults (0.90 threshold, 200000 limit)
- Existing sessions without context tracking continue normally
- ContextPressureError is caught internally, never bubbles to orchestrator

## Testing & Validation

### Unit Tests
- `ContextUsage.pressure_ratio()` calculation
- `parse_checkpoint_block()` parsing logic
- Lifecycle effect enum additions

### Integration Tests
- Full restart flow with mocked SDK
- Multiple sequential restarts
- Threshold edge cases (89%, 90%, 91%)

### Manual Verification
- Lower threshold to 1000 tokens locally
- Verify restart occurs and work continues
- Check logs show continuation_count

## Rollback Strategy

1. **Quick Disable**: Set `context_restart_threshold` to 1.1 (never triggers)
2. **Full Revert**: Revert the PR
3. **Verify**: Run test suite, confirm existing behavior restored

No external state, no database changes, no file persistence - rollback is trivial.

