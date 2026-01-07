# Agent SDK Reviewer Integration

**Tier:** M (Medium - Small feature with parsing, config, and integration)
**Owner:** Core Engineering
**Target ship:** Next Release
**Links:** [Project Config Docs](docs/project-config.md)

## 1. Outcome & Scope

**Problem / context**
Currently, mala relies on Cerberus (via the `review-gate` CLI) for code review/validation. This requires users to install an external binary/plugin, creating high friction for new users who want to run mala out-of-the-box.

**Goal**
Enable users to run mala without installing Cerberus by making an in-process Claude Agent SDK reviewer the default, while retaining Cerberus as a configurable override for users who prefer it.

**Success criteria**
- A new user can clone mala, install Python dependencies, and run a successful review cycle without installing `review-gate`
- Existing users can switch between `agent-sdk` and `cerberus` via `mala.yaml`
- The Agent SDK reviewer matches the `ReviewResult` format of Cerberus, ensuring pipeline compatibility
- Agent SDK reviewer runs in-process (same Python process), not as external subprocess

**Non-goals**
- Running both Cerberus and Agent SDK reviewers simultaneously (user chooses one via config)
- Removing Cerberus support (it remains as an optional override)
- Verdict aggregation between multiple reviewer types (only one reviewer type runs at a time)
- Multiple parallel reviewers within Agent SDK (single Claude session)
- Mode selection (fast/smart/max) for Agent SDK reviewer (always use default model)
- Changing the core `CodeReviewer` protocol signature

## 2. User Experience & Flows

**Primary flow**
1. User runs mala orchestrator on an issue (e.g., `mala run <issue-id>`)
2. Orchestrator factory reads `mala.yaml`. If `reviewer_type` is missing or set to `agent-sdk`, system initializes `AgentSDKReviewer`
3. When validation is triggered, `AgentSDKReviewer` constructs a prompt with the diff and context
4. The request is sent to Claude via the Agent SDK (in-process, same Python process)
5. The response is parsed into a `ReviewResult` (Pass/Fail/Needs Work + Findings)
6. The orchestrator acts on the result (commits or requests fixes) exactly as it did with Cerberus

**Key states**
- **Default:** No config needed; Agent SDK is used by default
- **Override:** User sets `reviewer_type: cerberus` in `mala.yaml` to use the legacy CLI
- **Reviewing:** Logs indicate "Reviewing with Claude Agent SDK..." instead of spawning a subprocess
- **Success:** Reviewer returns `ReviewResult` object with verdict (PASS/FAIL/NEEDS_WORK) and findings
- **Error state(s):**
  - API keys missing or SDK fails → review fails gracefully with descriptive error
  - Timeout waiting for reviewer → return `ReviewResult(passed=False, parse_error="timeout", fatal_error=False)`
  - Agent SDK raises exception during review → return `ReviewResult(passed=False, fatal_error=False, parse_error="...")`
  - Invalid review output (missing required fields) → return `ReviewResult(passed=False, fatal_error=False, parse_error="invalid format")`

## 3. Requirements + Verification

**R1 — Protocol Compatibility**
- **Requirement:** The Agent SDK reviewer MUST implement the `CodeReviewer` protocol with signature: `async def __call__(diff_range, context_file=None, timeout=300, claude_session_id=None, *, commit_shas=None) -> ReviewResultProtocol`
- **Empty diff handling:** The reviewer MUST check if the diff is empty and return `ReviewResult(passed=True, issues=[])` immediately without calling the Agent SDK. This applies to:
  - Range-based reviews: Check `git diff {diff_range}` output is non-empty before reviewing
  - Commit-based reviews: Check `git show {commit_sha}` output is non-empty for each commit before reviewing
  - If all diffs are empty, short-circuit to PASS
- **Verification:**
  - Given a `ReviewRunner` configured with Agent SDK reviewer
  - When `code_reviewer(diff_range="main..HEAD", context_file="/tmp/ctx.md", timeout=300)` is called
  - Then it returns a `ReviewResult` object satisfying `ReviewResultProtocol` (passed, issues, parse_error, fatal_error, review_log_path)
  - Given `diff_range` produces empty diff (no changes)
  - When `code_reviewer(diff_range="HEAD..HEAD")` is called
  - Then it returns `ReviewResult(passed=True, issues=[])` without calling Agent SDK
  - Given `commit_shas=["abc123"]` where abc123 has no diff changes
  - When `code_reviewer(commit_shas=["abc123"])` is called
  - Then it returns `ReviewResult(passed=True, issues=[])` without calling Agent SDK

**R2 — Default Configuration & Override**
- **Requirement:** The system MUST use `AgentSDKReviewer` by default if no configuration is present. It MUST use `CerberusReviewer` if `reviewer_type: cerberus` is specified in `mala.yaml`.
- **Configuration schema:** Add to `mala.yaml` (top-level field):
  ```yaml
  # Code reviewer configuration (optional)
  # Default: agent_sdk (in-process Claude Agent SDK reviewer)
  # Options: agent_sdk, cerberus
  reviewer_type: agent_sdk  # or cerberus

  # Agent SDK reviewer timeout (optional, default: 600s - longer for interactive exploration)
  agent_sdk_review_timeout: 600

  # Agent SDK reviewer model (optional, default: sonnet)
  agent_sdk_reviewer_model: sonnet
  ```
- **Schema location:** Document in `docs/project-config.md` under "Code Review Configuration" section
- **Verification:**
  - Given `mala.yaml` has no `reviewer_type` config
  - When orchestrator is created via factory
  - Then `AgentSDKReviewer` is instantiated and logs show "Initializing code reviewer: agent-sdk"
  - Given `mala.yaml` contains `reviewer_type: cerberus` and Cerberus plugin is installed
  - When orchestrator is created via factory
  - Then `CerberusReviewer` (subprocess-based) is instantiated and logs show "Initializing code reviewer: cerberus"

**R3 — Prompt & Context Parity**
- **Requirement:** The reviewer MUST use the established review system prompt (aligned with Cerberus) and inject the git diff and provided context file content into the user message.
- **Verification:**
  - Given Agent SDK reviewer is invoked with diff_range="main..HEAD" and context_file="/tmp/ctx.md"
  - When the review runs (inspect debug logs or mock capture)
  - Then the system prompt matches the source of truth (bundled in `src/domain/prompts.py` or equivalent)
  - And the user message contains both git diff content and context file content

**R4 — Output Normalization**
- **Requirement:** The reviewer MUST parse the LLM response into a `ReviewResult` object containing a `consensus_verdict` (PASS/FAIL/NEEDS_WORK) and a list of `ReviewIssue` objects (file, line, priority, description) if issues exist.
- **Verification:**
  - Given Agent SDK returns JSON: `{"verdict": "FAIL", "findings": [{"file_path": "src/main.py", "line_start": 42, "line_end": 42, "priority": 1, "title": "[P1] Issue", "body": "Details"}]}`
  - When the response is parsed
  - Then it produces `ReviewResult(passed=False, issues=[ReviewIssue(file="src/main.py", line_start=42, line_end=42, priority=1, title="[P1] Issue", body="Details", reviewer="agent-sdk")])`
  - Given malformed JSON or missing required fields
  - When the response is parsed
  - Then it returns `ReviewResult(passed=False, parse_error="...", fatal_error=False)`

**R5 — Timeout Management**
- **Requirement:** The reviewer MUST enforce the `timeout` parameter (default 600s for Agent SDK - longer to allow interactive exploration), cancelling the Agent SDK request if exceeded.
- **Verification:**
  - Given no timeout specified in call
  - When Agent SDK reviewer is invoked
  - Then it uses 600s timeout (or configured `agent_sdk_review_timeout`)
  - Given a mocked slow Agent SDK response exceeding timeout
  - When the review runs
  - Then it returns `ReviewResult(passed=False, parse_error="timeout", fatal_error=False)` after timeout

**R6 — In-Process Execution**
- **Requirement:** The review MUST run within the main Python process using the installed `anthropic` / Agent SDK libraries, NOT via a subprocess call to an external binary.
- **Verification:**
  - Given Agent SDK reviewer is invoked
  - When the review runs
  - Then no `review-gate` child process is spawned (verify via process monitoring)
  - And direct library usage is confirmed (code review or static analysis)

**R7 — overrides_disabled_setting Behavior**
- **Requirement:** The Agent SDK reviewer MUST implement `overrides_disabled_setting()` returning `False` (respects review disabled setting).
- **Verification:**
  - Given Agent SDK reviewer instance
  - When `overrides_disabled_setting()` is called
  - Then it returns `False`

## 4. Instrumentation & Release Checks

**Instrumentation**
- Log `Info`: "Initializing code reviewer: [Type]" at startup
- Log `Info`: "Starting review for [diff_range] with [Type]" when triggered
- Log `Debug`: Full prompt and raw response for debugging (behind debug flag)
- Log `Error`: Stack traces for SDK exceptions (auth, connection, timeout)
- Events to track via existing `MalaEventSink` (following DefaultReviewer pattern):
  - `on_review_warning()` - For any parse issues, timeouts, or non-fatal errors
  - Note: Full telemetry (on_review_started, on_review_passed, on_review_retry) is handled at the orchestrator level, not within the reviewer. The reviewer only emits warnings for non-fatal errors.

**Launch checklist**
- [ ] All R1-R7 requirements verified with tests
- [ ] Unit tests for `AgentSDKReviewer` with mocked SDK client
- [ ] Integration test with `ReviewRunner` using Agent SDK reviewer
- [ ] E2E test (optional): Real Claude call with small diff
- [ ] Agent SDK reviewer works without Cerberus installed
- [ ] Cerberus override config works correctly
- [ ] Timeout handling tested (both normal completion and timeout)
- [ ] Error states covered (SDK exception, invalid output, timeout)
- [ ] Empty diff short-circuits to PASS (matching Cerberus behavior)
- [ ] Existing tests pass when Agent SDK is default

**Decisions made**
1. **Review prompt location:** Create new file `src/prompts/review_agent.md` containing the agent-friendly review prompt with tool usage guidance (based on Cerberus review-code.md but adapted for agent exploration)
2. **Review prompt content:** The prompt must instruct Claude to:
   - Review the provided git diff for code quality, correctness, and adherence to project standards
   - Output JSON matching the schema defined in "Agent SDK Review Output Schema" section
   - Use P0-P3 priority levels (P0=blocker, P1=major, P2=should fix, P3=nit)
   - Include file paths, line numbers, titles, and detailed explanations for each finding
   - Return verdict "PASS" if no issues, "FAIL" for blocking issues (P0/P1), "NEEDS_WORK" for non-blocking issues (P2/P3)
3. **Timeout default:** 600s for Agent SDK (agents need time for interactive exploration; configurable via `agent_sdk_review_timeout`)
4. **Priority schema:** Use P0-P3 (int 0-3) to match Cerberus; "agent-sdk" as reviewer name
5. **Protocol compatibility:** Full compatibility with CodeReviewer protocol
6. **File structure:** New file `src/infra/clients/agent_sdk_review.py` following same pattern as `cerberus_review.py`
7. **Default behavior:** Agent SDK is default, Cerberus is optional override via `reviewer_type: cerberus` in `mala.yaml`
8. **In-process execution:** Agent SDK runs in same Python process, no subprocess
9. **Authentication:** Reuse the existing `ANTHROPIC_API_KEY` environment variable used by the rest of mala
10. **Diff delivery:** Use Agent SDK with tools. The agent has access to git and file reading tools via the `claude_code` preset. The reviewer passes the diff_range in the query, and the agent uses its tools to run `git diff`, read files, and explore the codebase interactively. This enables more thorough reviews where the agent can examine context beyond the immediate diff.

**Open questions (resolved)**
1. ~~How should Agent SDK reviewer format its output internally?~~ Prompt instructs Claude to output JSON; parse to ReviewResult
2. ~~Cache review prompt?~~ N/A since bundled in codebase
3. ~~Cerberus prompt not found?~~ N/A since using bundled prompt
4. ~~Mode support (fast/smart/max)?~~ No mode support; always use default model
5. ~~Agent SDK dependency handling?~~ Agent SDK is required dependency of mala (already used for agent sessions)
6. ~~Create artifacts on disk?~~ No disk artifacts (simpler than Cerberus); log via standard logging
7. ~~Migration docs?~~ Not needed; Cerberus users continue unchanged
8. ~~Feature flag to disable reviews?~~ Already exists via `--disable-validations review`

**Agent SDK Review Output Schema**

The Agent SDK reviewer must instruct Claude to output JSON matching this schema:

```json
{
  "verdict": "PASS" | "FAIL" | "NEEDS_WORK",
  "findings": [
    {
      "file_path": "relative/path/to/file.py",
      "line_start": 42,
      "line_end": 45,
      "priority": 0 | 1 | 2 | 3,
      "title": "[P0-3] Brief issue description",
      "body": "Detailed explanation of the issue and suggested fix"
    }
  ]
}
```

**Field requirements:**
- `verdict` (required, string): Must be exactly "PASS", "FAIL", or "NEEDS_WORK"
- `findings` (required, array): Empty array for PASS, non-empty for FAIL/NEEDS_WORK
- `file_path` (required in finding, string): Relative path from repo root
- `line_start` (required in finding, integer): Starting line number (1-indexed)
- `line_end` (required in finding, integer): Ending line number (≥ line_start)
- `priority` (required in finding, integer): 0=P0 (blocker), 1=P1 (major), 2=P2 (should fix), 3=P3 (nit)
- `title` (required in finding, string): Must start with "[P0]", "[P1]", "[P2]", or "[P3]"
- `body` (required in finding, string): Detailed explanation

**Parsing behavior:**
- Missing required fields → `ReviewResult(passed=False, parse_error="missing field: {field_name}", fatal_error=False)`
- Invalid verdict value → `ReviewResult(passed=False, parse_error="invalid verdict: {value}", fatal_error=False)`
- Malformed JSON → `ReviewResult(passed=False, parse_error="invalid json: {error}", fatal_error=False)`

**Open questions (for implementation)**
1. ~~What is the exact JSON schema for Claude's review output?~~ **RESOLVED:** Schema defined above
2. Should we use streaming for Agent SDK response or single query? (Single query for simplicity - no streaming)
