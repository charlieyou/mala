# ⚠️ DEPRECATED

**This plan is obsolete.** It was superseded by the v2 plan which uses the Claude Agent SDK instead of the Messages API.

**See instead:**
- **Authoritative Plan**: [2026-01-07-agent-sdk-reviewer-v2-plan.md](./2026-01-07-agent-sdk-reviewer-v2-plan.md)
- **Spec**: [2026-01-07-agent-sdk-reviewer-spec.md](./2026-01-07-agent-sdk-reviewer-spec.md)

**Key differences from this plan:**
- v2 uses `claude.Agent` from the Claude Agent SDK (not `anthropic.messages.create`)
- v2 uses `ClaudeCodeReviewer` class name (not `AgentSDKReviewer`)
- v2 provides read-only filesystem tools to the agent for context gathering
- v2 has different prompt structure and JSON schema

---

# ~~Implementation Plan: Agent SDK Reviewer Integration~~ (DEPRECATED)

## Context & Goals
- **Spec**: `/home/cyou/mala/plans/2026-01-07-agent-sdk-reviewer-spec.md`
- **Goal**: Implement `AgentSDKReviewer` as the default in-process code reviewer for `mala`, replacing the external Cerberus process.
- **Value**: Reduces installation friction (no Cerberus plugin required) and enables faster, more integrated code reviews using the Claude Agent SDK directly.

## Scope & Non-Goals
- **In Scope**
  - Implement `AgentSDKReviewer` class adhering to the `CodeReviewer` protocol.
  - Create a new review system prompt (`src/prompts/review.md`) based on the existing Cerberus prompt.
  - Update `ValidationConfig` to support a `reviewer_type` setting (default: `agent_sdk`, fallback: `cerberus`).
  - Update `Factory` to instantiate the appropriate reviewer based on config.
  - Add unit, integration, and optional E2E tests for the new reviewer.
- **Out of Scope (Non-Goals)**
  - Changing the behavior of the existing `DefaultReviewer` (Cerberus).
  - Adding new review capabilities beyond parity with Cerberus.
  - Customizing the prompt content beyond the initial port.

## Assumptions & Constraints
- **Assumptions**
  - The `anthropic` SDK is available in the environment (implied by `src/infra/sdk_adapter.py`).
  - `src/infra/tools/command_runner.py` provides necessary git operations (diff, show).
  - The JSON schema for review output is fixed as per spec (Verdict, Priority P0-P3, Findings).

### Implementation Constraints
- **Pattern Matching**: Must use `@dataclass` for the reviewer, similar to `DefaultReviewer`.
- **Protocol**: Must strictly adhere to `src/core/protocols.py:CodeReviewer`.
- **Dependencies**: Use `anthropic.Anthropic` client directly (SDK adapter is for Agent SDK, not API SDK).
- **Error Handling**: Distinguish between fatal errors (config) and non-fatal errors (API timeouts), returning appropriate `ReviewResult` states.

### Testing Constraints
- **Unit Tests**: Must mock the Anthropic client to avoid network calls and costs.
- **E2E Tests**: Mark with `@pytest.mark.e2e` to prevent accidental execution in CI without credentials.
- **Coverage**: Ensure handling of empty diffs, malformed JSON responses, and API timeouts.

## Prerequisites
- [ ] `ANTHROPIC_API_KEY` available in the environment for E2E testing.
- [ ] `src/prompts/review.md` content available (to be copied/adapted from Cerberus).

## High-Level Approach

### Phase 1: Infrastructure & Config
**Files to modify/create:**
- **New**: `src/prompts/review.md` - Review system prompt based on Cerberus review-code.md
- **Modify**: `src/domain/validation/config.py` - Add `reviewer_type: Literal["agent_sdk", "cerberus"]` field to `ValidationConfig`
- **Modify**: `src/domain/prompts.py` - Add `review_prompt: str` field to `PromptProvider` dataclass
- **Modify**: `src/domain/prompts.py` - Update `load_prompts()` function to include review.md

**Specific changes:**
1. Create `src/prompts/review.md` with content:
   - **Source reference**: Copy from `/home/cyou/.claude/plugins/cache/cerberus/cerberus/*/prompts/generators/review-code.md`
   - **Adapt for Agent SDK**: Ensure JSON schema uses int priorities (0-3), not strings ("P0"-"P3")
   - **Content**:
     - Instructions to review code for quality, correctness, adherence to standards
     - JSON output schema matching spec (verdict: PASS/FAIL/NEEDS_WORK, findings array with int priorities)
     - P0-P3 priority definitions (P0=blocker, P1=major, P2=should fix, P3=nit)
     - Focus areas: correctness, security, performance, maintainability, test coverage, breaking changes

2. Add to `ValidationConfig` dataclass (after line 592):
   ```python
   reviewer_type: Literal["agent_sdk", "cerberus"] = "agent_sdk"
   agent_sdk_review_timeout: int = 300
   agent_sdk_reviewer_model: str = "claude-3-5-sonnet-20241022"  # Configurable model
   ```

3. Add to `PromptProvider` dataclass (after line 87) with default for backwards compatibility:
   ```python
   review_prompt: str = ""  # Default empty string for backwards compatibility
   ```
   **Note**: All existing instantiation sites will continue to work. The factory will provide actual content.

4. Update `load_prompts()` function (after line 110):
   ```python
   review_prompt=(prompt_dir / "review.md").read_text(),
   ```

### Phase 2: AgentSDKReviewer Implementation
**File to create:**
- **New**: `src/infra/clients/agent_sdk_review.py` - Main reviewer implementation

**Class structure (dataclass pattern matching DefaultReviewer):**
```python
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.infra.clients.review_output_parser import ReviewIssue, ReviewResult
from src.infra.tools.command_runner import CommandRunner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from src.core.protocols import MalaEventSink


@dataclass
class AgentSDKReviewer:
    """Agent SDK-based code reviewer implementing CodeReviewer protocol."""

    repo_path: Path
    review_prompt: str  # Loaded once in factory
    event_sink: MalaEventSink | None = None
    model: str = "claude-3-5-sonnet-20241022"
    default_timeout: int = 300

    def overrides_disabled_setting(self) -> bool:
        """Return False; AgentSDKReviewer respects the disabled setting."""
        return False

    async def __call__(
        self,
        diff_range: str,
        context_file: Path | None = None,
        timeout: int = 300,
        claude_session_id: str | None = None,
        *,
        commit_shas: Sequence[str] | None = None,
    ) -> ReviewResult:
        """Run code review using Claude Agent SDK."""
        # Implementation below
```

**Implementation steps for `__call__` method:**

1. **Empty diff check** (short-circuit to PASS):
   ```python
   runner = CommandRunner(cwd=self.repo_path)

   # Check for empty diff before calling SDK
   if not commit_shas:
       if await self._check_diff_empty(diff_range, runner):
           return ReviewResult(
               passed=True, issues=[], parse_error=None,
               fatal_error=False, review_log_path=None
           )
   ```

2. **Get diff content**:
   ```python
   try:
       diff_content = await self._get_diff_content(
           diff_range, commit_shas, runner
       )
   except Exception as e:
       return ReviewResult(
           passed=False, issues=[],
           parse_error=f"Failed to get diff: {e}",
           fatal_error=False, review_log_path=None
       )
   ```

3. **Load context file if provided** (using async I/O):
   ```python
   import asyncio

   context_content = ""
   if context_file and context_file.exists():
       # Use asyncio.to_thread for non-blocking file I/O
       context_content = await asyncio.to_thread(context_file.read_text)
   ```

4. **Construct prompt**:
   ```python
   user_message = f"# Git Diff\n\n{diff_content}"
   if context_content:
       user_message += f"\n\n# Context\n\n{context_content}"
   ```

5. **Call Anthropic API** (using async client):
   ```python
   import anthropic

   # Use AsyncAnthropic for non-blocking async I/O
   client = anthropic.AsyncAnthropic()  # Uses ANTHROPIC_API_KEY env var

   # Emit review started event if event_sink is available
   if self.event_sink:
       self.event_sink.on_review_started(
           agent_id=claude_session_id,
           attempt=1,
           max_attempts=1
       )

   try:
       response = await client.messages.create(
           model=self.model,
           max_tokens=4096,
           system=self.review_prompt,
           messages=[{"role": "user", "content": user_message}],
           timeout=timeout,
       )

       # Extract text from response
       response_text = response.content[0].text

   except anthropic.APITimeoutError:
       if self.event_sink:
           self.event_sink.on_review_warning(
               message=f"Review timed out after {timeout}s",
               agent_id=claude_session_id
           )
       return ReviewResult(
           passed=False, issues=[], parse_error="timeout",
           fatal_error=False, review_log_path=None
       )
   except anthropic.AuthenticationError as e:
       # Fatal error: missing or invalid API key
       return ReviewResult(
           passed=False, issues=[],
           parse_error=f"Authentication failed: {e}",
           fatal_error=True, review_log_path=None
       )
   except Exception as e:
       return ReviewResult(
           passed=False, issues=[],
           parse_error=f"API error: {e}",
           fatal_error=False, review_log_path=None
       )
   ```

6. **Parse response and emit telemetry**:
   ```python
   result = self._parse_response(response_text)

   # Emit telemetry after parsing (not inside _parse_response to avoid scope issues)
   if self.event_sink and not result.parse_error:
       if result.passed:
           self.event_sink.on_review_passed(agent_id=claude_session_id)
       elif result.issues:
           self.event_sink.on_review_retry(
               agent_id=claude_session_id,
               attempt=1,
               max_attempts=1,
               error_count=len(result.issues),
               parse_error=None
           )

   return result
   ```

**Helper methods to implement:**

1. `async def _check_diff_empty(self, diff_range: str, runner: CommandRunner) -> bool`:
   - Run `git diff --stat {diff_range}`
   - Return True if stdout is empty
   - Reuse pattern from `cerberus_gate_cli.py:check_diff_empty` (lines 305-330)

2. `async def _get_diff_content(self, diff_range: str, commit_shas: Sequence[str] | None, runner: CommandRunner) -> str`:
   - If commit_shas: run `git show {sha}` for each commit, concatenate
   - Else: run `git diff {diff_range}`
   - Return concatenated diff content

3. `def _parse_response(self, response_text: str) -> ReviewResult`:
   - Extract JSON from response (may be wrapped in ```json blocks)
   - Parse JSON and validate schema
   - Map verdict to passed boolean: PASS → True, FAIL/NEEDS_WORK → False
   - Create ReviewIssue objects from findings array
   - Return ReviewResult with parsed data
   - On error: return ReviewResult with parse_error

**JSON parsing logic:**
```python
def _parse_response(self, response_text: str) -> ReviewResult:
    import json
    import re

    # Extract JSON from code blocks if present (robust extraction)
    # Try 1: Match ```json ... ``` blocks
    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response_text, re.DOTALL)
    if json_match:
        json_text = json_match.group(1).strip()
    else:
        # Try 2: Find first { and last } (fallback for unformatted responses)
        first_brace = response_text.find('{')
        last_brace = response_text.rfind('}')
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            json_text = response_text[first_brace:last_brace+1].strip()
        else:
            json_text = response_text.strip()

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        return ReviewResult(
            passed=False, issues=[],
            parse_error=f"Invalid JSON: {e}",
            fatal_error=False, review_log_path=None
        )

    # Validate required fields
    if "verdict" not in data:
        return ReviewResult(
            passed=False, issues=[],
            parse_error="Missing 'verdict' field",
            fatal_error=False, review_log_path=None
        )

    verdict = data["verdict"]
    if verdict not in ("PASS", "FAIL", "NEEDS_WORK"):
        return ReviewResult(
            passed=False, issues=[],
            parse_error=f"Invalid verdict: {verdict}",
            fatal_error=False, review_log_path=None
        )

    findings = data.get("findings", [])
    issues = []

    for finding in findings:
        try:
            # Convert priority: int (0-3) from JSON matches ReviewIssue schema
            # If priority is string like "P1", convert to int
            priority_raw = finding["priority"]
            if isinstance(priority_raw, str) and priority_raw.startswith("P"):
                priority = int(priority_raw[1:])  # "P1" -> 1
            else:
                priority = int(priority_raw)  # Already int

            issues.append(ReviewIssue(
                file=finding["file_path"],
                line_start=finding["line_start"],
                line_end=finding["line_end"],
                priority=priority,  # Now correctly int (0-3)
                title=finding["title"],
                body=finding["body"],
                reviewer="agent-sdk"
            ))
        except (KeyError, TypeError, ValueError) as e:
            return ReviewResult(
                passed=False, issues=[],
                parse_error=f"Invalid finding format: {e}",
                fatal_error=False, review_log_path=None
            )

    passed = (verdict == "PASS")

    # No telemetry here - telemetry is emitted in __call__ after parsing
    # to avoid scope issues with claude_session_id

    return ReviewResult(
        passed=passed,
        issues=issues,
        parse_error=None,
        fatal_error=False,
        review_log_path=None
    )
```

### Phase 3: Factory Integration
**File to modify:**
- **Modify**: `src/orchestration/factory.py` - Update `create_orchestrator()` to select reviewer

**Specific changes:**

1. Add import at top (around line 207):
   ```python
   from src.infra.clients.agent_sdk_review import AgentSDKReviewer
   ```

2. Read reviewer_type from config (after loading mala_config, around line 250):
   ```python
   reviewer_type = getattr(mala_config, "reviewer_type", "agent_sdk")
   ```

3. Replace reviewer creation logic (around lines 280-295):
   ```python
   # Code reviewer
   code_reviewer: CodeReviewer
   if deps is not None and deps.code_reviewer is not None:
       code_reviewer = deps.code_reviewer
   else:
       # Select reviewer based on config
       if reviewer_type == "cerberus":
           code_reviewer = cast(
               "CodeReviewer",
               DefaultReviewer(
                   repo_path=repo_path,
                   bin_path=mala_config.cerberus_bin_path,
                   spawn_args=mala_config.cerberus_spawn_args,
                   wait_args=mala_config.cerberus_wait_args,
                   env=mala_config.cerberus_env,
                   event_sink=event_sink,
               ),
           )
       else:  # Default to agent_sdk
           agent_sdk_timeout = getattr(mala_config, "agent_sdk_review_timeout", 300)
           agent_sdk_model = getattr(mala_config, "agent_sdk_reviewer_model", "claude-3-5-sonnet-20241022")
           code_reviewer = cast(
               "CodeReviewer",
               AgentSDKReviewer(
                   repo_path=repo_path,
                   review_prompt=prompts.review_prompt,
                   event_sink=event_sink,
                   model=agent_sdk_model,  # Use configurable model from config
                   default_timeout=agent_sdk_timeout,
               ),
           )
   ```

### Phase 4: Testing

**Unit tests** (`tests/unit/infra/test_agent_sdk_review.py`):

Create comprehensive unit tests with mocked Anthropic client:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from src.infra.clients.agent_sdk_review import AgentSDKReviewer
from src.infra.clients.review_output_parser import ReviewResult


class TestAgentSDKReviewer:
    @pytest.fixture
    def reviewer(self, tmp_path):
        return AgentSDKReviewer(
            repo_path=tmp_path,
            review_prompt="Review this code",
            event_sink=None,
        )

    @pytest.mark.asyncio
    async def test_empty_diff_returns_pass(self, reviewer):
        """Empty diff should short-circuit to PASS without calling SDK."""
        # Mock git diff --stat returning empty
        # Verify ReviewResult(passed=True, issues=[])
        # Verify no Anthropic API call made

    @pytest.mark.asyncio
    async def test_successful_review_pass(self, reviewer):
        """Valid PASS verdict should parse correctly."""
        # Mock Anthropic API returning {"verdict": "PASS", "findings": []}
        # Verify ReviewResult(passed=True, issues=[])

    @pytest.mark.asyncio
    async def test_successful_review_fail(self, reviewer):
        """Valid FAIL verdict with findings should parse correctly."""
        # Mock Anthropic API returning FAIL with findings
        # Verify ReviewResult(passed=False, issues=[...])
        # Verify ReviewIssue fields match input

    @pytest.mark.asyncio
    async def test_malformed_json_returns_error(self, reviewer):
        """Invalid JSON should return parse_error."""
        # Mock Anthropic API returning "not json"
        # Verify ReviewResult(passed=False, parse_error="Invalid JSON: ...")

    @pytest.mark.asyncio
    async def test_missing_verdict_field(self, reviewer):
        """Missing verdict field should return parse_error."""
        # Mock Anthropic API returning {"findings": []}
        # Verify ReviewResult(parse_error="Missing 'verdict' field")

    @pytest.mark.asyncio
    async def test_invalid_verdict_value(self, reviewer):
        """Invalid verdict value should return parse_error."""
        # Mock Anthropic API returning {"verdict": "UNKNOWN", "findings": []}
        # Verify ReviewResult(parse_error="Invalid verdict: UNKNOWN")

    @pytest.mark.asyncio
    async def test_api_timeout(self, reviewer):
        """API timeout should return timeout parse_error."""
        # Mock Anthropic API raising APITimeoutError
        # Verify ReviewResult(parse_error="timeout")

    @pytest.mark.asyncio
    async def test_api_exception(self, reviewer):
        """API exception should return SDK error parse_error."""
        # Mock Anthropic API raising generic exception
        # Verify ReviewResult(parse_error="SDK error: ...")

    @pytest.mark.asyncio
    async def test_context_file_included(self, reviewer, tmp_path):
        """Context file content should be included in prompt."""
        # Create context file
        # Mock Anthropic API call
        # Verify user message contains both diff and context

    def test_overrides_disabled_setting_returns_false(self, reviewer):
        """overrides_disabled_setting should return False."""
        assert reviewer.overrides_disabled_setting() is False
```

**Integration tests** (extend `tests/integration/test_factory_wiring.py`):

```python
def test_factory_creates_agent_sdk_reviewer_by_default(tmp_path):
    """Factory should create AgentSDKReviewer when no reviewer_type config."""
    # Create minimal mala.yaml without reviewer_type
    # Call create_orchestrator
    # Assert code_reviewer is instance of AgentSDKReviewer


def test_factory_creates_cerberus_reviewer_when_configured(tmp_path):
    """Factory should create DefaultReviewer when reviewer_type=cerberus."""
    # Create mala.yaml with reviewer_type: cerberus
    # Call create_orchestrator
    # Assert code_reviewer is instance of DefaultReviewer
```

**E2E tests** (`tests/e2e/test_agent_sdk_e2e.py`):

```python
import pytest
from pathlib import Path

from src.infra.clients.agent_sdk_review import AgentSDKReviewer


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_real_agent_sdk_review(tmp_path):
    """Real API call with small diff (requires ANTHROPIC_API_KEY)."""
    # Skip if ANTHROPIC_API_KEY not set
    # Create temp git repo with small change
    # Create AgentSDKReviewer
    # Call reviewer with diff_range
    # Assert ReviewResult structure is valid
    # Assert verdict is one of PASS/FAIL/NEEDS_WORK
```

## Technical Design

### Architecture
The `AgentSDKReviewer` will sit alongside `DefaultReviewer` (Cerberus wrapper). The `Factory` determines which one to instantiate based on `mala.yaml`.

**Data Flow:**
1. `Orchestrator` requests review → calls `AgentSDKReviewer.__call__`.
2. `AgentSDKReviewer` calls `git diff` via `CommandRunner` to get diff content.
3. `AgentSDKReviewer` constructs prompt (System: `review.md`, User: Diff + Context).
4. `AgentSDKReviewer` calls `anthropic.Anthropic().messages.create()` with timeout.
5. Parse JSON response → `ReviewResult`.
6. Return to orchestrator.

### Data Model
- **Config**: Add `reviewer_type: Literal["agent_sdk", "cerberus"]` and `agent_sdk_review_timeout: int` to `ValidationConfig`.
- **Output**: Reuses existing `ReviewResult` and `ReviewIssue` from `src/infra/clients/review_output_parser.py`.

### API/Interface Design
**`AgentSDKReviewer`** (New Class)
- **Signature**: Matches `CodeReviewer` protocol exactly.
- **Constructor**:
  - `repo_path: Path` - Repository path for git operations
  - `review_prompt: str` - Loaded system prompt (loaded once in factory)
  - `event_sink: MalaEventSink | None` - Optional event sink for telemetry
  - `model: str` - Claude model ID (default: "claude-3-5-sonnet-20241022")
  - `default_timeout: int` - Default timeout in seconds (default: 300)

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/domain/validation/config.py` | Modify | Add `reviewer_type` and `agent_sdk_review_timeout` fields to `ValidationConfig` |
| `src/domain/prompts.py` | Modify | Add `review_prompt` to `PromptProvider`, update `load_prompts()` |
| `src/infra/clients/agent_sdk_review.py` | **New** | Implementation of `AgentSDKReviewer` |
| `src/prompts/review.md` | **New** | System prompt for code review |
| `src/orchestration/factory.py` | Modify | Logic to switch reviewer implementation based on config |
| `tests/unit/infra/test_agent_sdk_review.py` | **New** | Unit tests for new reviewer |
| `tests/integration/test_factory_wiring.py` | Modify | Add tests for reviewer selection |
| `tests/e2e/test_agent_sdk_e2e.py` | **New** | E2E test with real API |

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes
- **Empty Diff**:
  - *Handling*: Check diff size before SDK call using `git diff --stat`. If empty, return "Passed" result immediately.
  - *Implementation*: Reuse pattern from `cerberus_gate_cli.py:check_diff_empty()`

- **Malformed JSON**:
  - *Handling*: Catch `JSONDecodeError`. Try extracting JSON from code blocks first. Return `ReviewResult` with `parse_error` (do not crash).

- **Context Length Exceeded**:
  - *Handling*: Catch Anthropic `BadRequestError` (or similar). Return `ReviewResult` with `parse_error="context length exceeded"`.

- **Timeout**:
  - *Handling*: Pass `timeout` parameter to Anthropic client. Catch `APITimeoutError`. Return `ReviewResult` with `parse_error="timeout"`.

- **Missing API Key**:
  - *Handling*: Anthropic client will raise exception. Catch and return `ReviewResult` with `fatal_error=True` and descriptive message.

### Breaking Changes & Compatibility
- **Default Change**: The default behavior changes from Cerberus to Agent SDK.
  - *Mitigation*: Users can set `reviewer_type: cerberus` in `mala.yaml` to revert.
  - *Compat*: The `CodeReviewer` protocol ensures the rest of the system is unaffected.

- **New Dependencies**: None - `anthropic` SDK already used by mala for agent sessions.

## Testing & Validation Strategy

### Unit Tests (`tests/unit/infra/test_agent_sdk_review.py`)
- **Mock Strategy**: Mock `anthropic.Anthropic` client using `unittest.mock.patch`
- **Fixtures**: Shared `reviewer` fixture with tmp_path repo
- **Coverage**: All error paths, empty diff, valid/invalid responses, timeout, context file

**Key test cases:**
- `test_empty_diff_returns_pass`: Verify `git diff --stat` returning empty leads to immediate PASS.
- `test_successful_review_pass`: Mock SDK response with valid JSON (PASS), verify `ReviewResult` parsing.
- `test_successful_review_fail`: Mock SDK response with FAIL + findings, verify issues array.
- `test_malformed_json_returns_error`: Mock invalid JSON, verify error handling (returns failure result, doesn't raise).
- `test_missing_verdict_field`: Mock JSON without verdict, verify parse_error.
- `test_invalid_verdict_value`: Mock JSON with invalid verdict, verify parse_error.
- `test_api_timeout`: Mock `APITimeoutError`, verify graceful failure with "timeout" parse_error.
- `test_api_exception`: Mock SDK exception, verify graceful failure with SDK error message.
- `test_context_file_included`: Verify context file content is included in user message.
- `test_overrides_disabled_setting_returns_false`: Verify method returns False.

### Integration Tests (`tests/integration/test_factory_wiring.py`)
- **Test factory logic**: Verify correct reviewer instantiation based on config
- **Test cases:**
  - Verify that setting `reviewer_type="cerberus"` produces `DefaultReviewer`.
  - Verify that setting `reviewer_type="agent_sdk"` (or default) produces `AgentSDKReviewer`.
  - Verify that `agent_sdk_review_timeout` config is passed to reviewer.

### E2E Tests (`tests/e2e/test_agent_sdk_e2e.py`)
- **Mark**: `@pytest.mark.e2e` to prevent CI execution without credentials
- **Requirements**: `ANTHROPIC_API_KEY` environment variable
- **Test case**:
  - `test_real_agent_sdk_review`: Create dummy file change, run `AgentSDKReviewer`, assert valid `ReviewResult` structure (Verdict exists, Issues list is a list).

### Acceptance Criteria Coverage
| Spec AC | Covered By |
|---------|------------|
| R1: Protocol Compatibility | `AgentSDKReviewer` class definition, Unit Tests (test_successful_review_*) |
| R1: Empty Diff Handling | `_check_diff_empty()` method, Unit Tests (test_empty_diff_returns_pass) |
| R2: Configurable (Default SDK) | `ValidationConfig` changes, Factory logic, Integration Tests (test_factory_creates_*) |
| R3: Prompt Parity | `src/prompts/review.md` creation, Manual review |
| R4: JSON Output Normalization | `_parse_response()` method, Unit Tests (test_successful_review_*, test_malformed_*) |
| R5: Timeout Handling | Anthropic client timeout parameter, Unit Tests (test_api_timeout) |
| R6: In-Process Execution | Architecture (no subprocess calls), Code Review, Integration Tests |
| R7: overrides_disabled_setting | Unit Tests (test_overrides_disabled_setting_returns_false) |

## Decisions Made (from Spec + Interview)

1. **Config location**: Add `reviewer_type` to `ValidationConfig` in `src/domain/validation/config.py` (top-level field)
2. **Git diff execution**: Use `CommandRunner` to run `git diff` and `git show` commands (reuse existing pattern from cerberus_gate_cli.py)
3. **Test coverage**: Unit tests (comprehensive) + integration tests with factory + E2E tests with real SDK (marked @pytest.mark.e2e)
4. **Review prompt**: Create new `src/prompts/review.md` based on Cerberus review-code.md
5. **Class structure**: Use `@dataclass` pattern matching `DefaultReviewer` for consistency
6. **Prompt loading**: Load once in factory via `load_prompts()`, pass to constructor
7. **Timeout**: Use Anthropic client timeout parameter (pass to `messages.create()`)
8. **Error handling**: Distinguish fatal (missing API key) vs non-fatal (timeout, parse) errors
9. **SDK client**: Use `anthropic.Anthropic` directly (not SDK adapter which is for Agent SDK, not API SDK)
10. **Default model**: Use "claude-3-5-sonnet-20241022" as default (configurable via constructor)

## Open Questions (Resolved)

All implementation questions have been resolved:
1. **Git utility location**: Use `CommandRunner` from `src/infra/tools/command_runner.py` (pattern matches cerberus_gate_cli.py)
2. **SDK client**: Use `anthropic.Anthropic().messages.create()` with timeout parameter
3. **Event sink usage**: Optional parameter, emit standard review events if provided (on_review_started, on_review_passed, etc.)
4. **ValidationConfig schema**: Add two optional fields with defaults (reviewer_type="agent_sdk", agent_sdk_review_timeout=300)
5. **Prompt provider update**: Add `review_prompt: str` field to PromptProvider dataclass, update load_prompts() function

## Next Steps

After this plan is approved:
1. Run `/create-tasks` to generate the task breakdown:
   - `--beads` → Beads issues with dependencies for parallel execution
   - (default) → TODO.md checklist for sequential tracking
2. Begin with **Phase 1: Infrastructure & Config** (no dependencies)
3. Then **Phase 2: AgentSDKReviewer Implementation** (depends on Phase 1)
4. Then **Phase 3: Factory Integration** (depends on Phase 1 & 2)
5. Finally **Phase 4: Testing** (depends on Phase 2 & 3)
