# Epic Verification Agent

## Overview

An agent that verifies all commits made for issues under an epic collectively meet the epic's acceptance criteria before allowing the epic to close. This prevents premature epic closure when child issues were completed individually but the overall epic goals remain unmet.

## Goals

- Ensure epic-level acceptance criteria are satisfied by cumulative child issue work
- Automatically create remediation issues when gaps are identified
- Block epic closure until verification passes or explicit human override
- Provide clear, actionable feedback on unmet criteria

## Non-Goals (Out of Scope)

- Code quality review (already handled by external review via Cerberus)
- Individual issue validation (already handled by QualityGate)
- Performance testing or load testing
- Manual test execution
- Changing how individual issues are processed

## User Stories

- As a project maintainer, I want epics to only close when all acceptance criteria are met, so that I can trust epic completion status.
- As a developer, I want clear feedback on which epic criteria aren't met, so that I know what work remains.
- As an orchestrator operator, I want automatic remediation issue creation, so that gaps are tracked and assigned.

## Technical Design

### Architecture

The epic verification agent sits between issue completion and epic closure in the orchestrator:

```
Issue Completion → close_async() → [Epic Verification Agent] → close_eligible_epics_async()
                                           ↓ (if fail)
                                    Create remediation issues
                                    Add as epic blockers
```

**Key Principle**: The verification logic is NOT part of `BeadsClient`. It wraps the epic closure flow at the orchestrator level, keeping `BeadsClient` focused on Beads CLI interaction.

### Key Components

- **`EpicVerifier`** (`src/epic_verifier.py`): Main verification orchestrator
  - Gathers epic data (acceptance criteria, linked spec)
  - Computes scoped diff from child issue commits only
  - Invokes verification model
  - Parses verdict and creates remediation issues

- **`EpicVerificationModel`** (Protocol in `src/protocols.py`): Model-agnostic interface
  - `verify(epic_criteria: str, diff: str, spec_content: str | None) -> EpicVerdict`
  - Initial implementation: Claude via SDK
  - Designed for future model swapping (Codex, Gemini, local models)

- **Data Types** (in `src/models.py` to avoid circular imports):
  - `EpicVerdict`: Structured verification result
  - `UnmetCriterion`: Individual gap detail
  - `EpicVerificationResult`: Summary of verification run

### Diff Scoping Strategy

**Problem**: A naive `git diff <baseline>..HEAD` includes unrelated commits (parallel epics, hotfixes), leading to false pass/fail.

**Solution**: Scope the diff to only commits linked to child issues:

1. Collect all child issue IDs via `bd dep tree <epic_id> --direction=up --json`
2. For each child, find commits matching `^bd-<child_id>:` in commit message
3. Generate a combined diff of only those commits:
   ```bash
   git diff-tree -p <commit1> <commit2> ... | git apply --stat
   # Or: concatenate individual commit diffs
   git show --format= <commit1> <commit2> ...
   ```
4. This produces a diff containing exactly the work done for the epic's children

**Handling edge cases**:
- **Squashed commits**: Match `bd-<child_id>:` prefix; squashing preserves the prefix
- **Rebased commits**: Same—match by message prefix, not commit SHA
- **Merge commits**: Skip merge commits (no `-m` diff), only include actual work commits
- **Multiple commits per issue**: Include all commits matching the issue prefix
- **No matching commits**: Block closure, require human review

### Baseline Computation (Deprecated)

The original "baseline" approach is replaced by the scoped diff strategy above. No single baseline commit is needed—instead, we aggregate the specific commits linked to child issues.

### Agent-Driven Exploration

**Approach**: The verification agent receives a commit list and range hint, then
autonomously explores the repository using tools (Bash for git commands, Glob,
Grep, Read) to inspect the relevant changes.

**Benefits**:
1. No context limit issues from large diffs
2. Agent can focus on specific areas relevant to acceptance criteria
3. More thorough verification through iterative exploration

**Configuration**:
- `timeout_ms`: Model timeout (default: 300000ms / 5 minutes)

### Data Model

Located in `src/models.py`:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class UnmetCriterion:
    """Individual gap identified during epic verification."""
    criterion: str      # The acceptance criterion not met
    evidence: str       # Why it's considered unmet
    severity: Literal["critical", "major", "minor"]
    criterion_hash: str  # SHA256 of criterion text, for deduplication

@dataclass
class EpicVerdict:
    """Result of verifying an epic against its acceptance criteria."""
    passed: bool
    unmet_criteria: list[UnmetCriterion]
    confidence: float  # 0.0 to 1.0
    reasoning: str

@dataclass
class EpicVerificationResult:
    """Summary of a verification run across multiple epics."""
    verified_count: int           # Number of epics verified
    passed_count: int             # Number that passed
    failed_count: int             # Number that failed
    human_review_count: int       # Number flagged for human review
    verdicts: dict[str, EpicVerdict]  # epic_id -> verdict
    remediation_issues_created: list[str]  # Issue IDs created
```

### Spec File Parsing

Epic descriptions may contain references to spec files:

```markdown
Implements the authentication system.

See specs/auth/login.md for detailed requirements.
```

The verifier extracts spec paths via regex patterns (case-insensitive):

```python
SPEC_PATH_PATTERNS = [
    r'[Ss]ee\s+(specs/[\w/.-]+\.(?:md|MD))',           # "See specs/foo/bar.md"
    r'[Ss]pec:\s*(specs/[\w/.-]+\.(?:md|MD))',         # "Spec: specs/foo.md"
    r'\[(specs/[\w/.-]+\.(?:md|MD))\]',                # "[specs/foo.md]"
    r'(?:^|\s)(specs/[\w/.-]+\.(?:md|MD))(?:\s|$)',    # Bare "specs/foo.md"
]
```

Supports:
- Nested directories: `specs/auth/v2/login.md`
- Case-insensitive extensions: `.md`, `.MD`
- Markdown link syntax: `[specs/foo.md]`

If found and file exists, content is included in verification context.

### API Design

```python
class EpicVerifier:
    def __init__(
        self,
        beads: BeadsClient,
        model: EpicVerificationModel,
        repo_path: Path,
        retry_config: RetryConfig | None = None,
        lock_manager: object | None = None,
        event_sink: EventSink | None = None,
    ): ...

    async def verify_and_close_eligible(
        self,
        human_override_epic_ids: set[str] | None = None,
    ) -> EpicVerificationResult:
        """
        Check for epics eligible to close, verify each, and close those that pass.

        Args:
            human_override_epic_ids: Epic IDs to close without verification
                                     (explicit human override)

        Returns:
            Summary of verification results.
        """
        ...

    async def verify_epic(self, epic_id: str) -> EpicVerdict:
        """Verify a single epic against its acceptance criteria."""
        ...

    async def create_remediation_issues(
        self, epic_id: str, verdict: EpicVerdict
    ) -> list[str]:
        """
        Create issues for unmet criteria, return their IDs.

        Deduplication: Checks for existing issues with matching
        `epic_remediation:<epic_id>:<criterion_hash>` tag before creating.
        """
        ...

    async def add_epic_blockers(
        self, epic_id: str, blocker_issue_ids: list[str]
    ) -> None:
        """
        Add issues as blockers of the epic via `bd dep add`.

        Args:
            epic_id: The epic to block
            blocker_issue_ids: Issues that must be resolved before epic closes
        """
        ...

    async def request_human_review(
        self, epic_id: str, reason: str, verdict: EpicVerdict | None = None
    ) -> str:
        """
        Create a human review issue that blocks the epic.

        Returns:
            Issue ID of the created review request.
        """
        ...
```

### Protocol for Model Abstraction

Located in `src/protocols.py`:

```python
class EpicVerificationModel(Protocol):
    async def verify(
        self,
        epic_criteria: str,
        diff: str,
        spec_content: str | None,
    ) -> EpicVerdict:
        """
        Verify if the diff satisfies the epic's acceptance criteria.

        Args:
            epic_criteria: The epic's acceptance criteria text
            diff: Scoped git diff of child issue commits only
            spec_content: Optional content of linked spec file

        Returns:
            Structured verdict with pass/fail and unmet criteria details
        """
        ...
```

### Retry Configuration

```python
@dataclass
class RetryConfig:
    max_retries: int = 2           # Total retry attempts
    initial_delay_ms: int = 1000   # First retry delay
    backoff_multiplier: float = 2.0  # Exponential backoff
    max_delay_ms: int = 30000      # Cap on retry delay
    timeout_ms: int = 120000       # Per-attempt timeout (2 min)
```

Aligned with existing `max_review_retries` pattern in orchestrator.

### Human Override Mechanism

**Triggering conditions**:
1. Missing acceptance criteria (no criteria to verify)
2. Model returns low confidence (<0.5)
3. Model verification timeout after retries exhausted
4. Cannot compute diff (no child commits found)

**Mechanism**:

1. **Create human review issue** via `bd issue create`:
   ```
   Title: "[Human Review] Epic mala-xyz requires manual verification"
   Description: <reason for review, verdict details if available>
   Priority: P1
   Tags: epic_human_review, epic_id:<epic_id>
   ```

2. **Add as epic blocker** via `bd dep add`:
   ```bash
   bd dep add mala-xyz --blocked-by <review_issue_id>
   ```

3. **Epic remains open** until human either:
   - Closes the review issue (signals approval) → epic becomes eligible again
   - Uses `--human-override` flag when running orchestrator:
     ```bash
     mala run --epic-override mala-xyz
     ```
     This passes the epic ID to `verify_and_close_eligible(human_override_epic_ids={...})`

4. **Logging**:
   ```
   ⚠ Epic mala-xyz flagged for human review: missing acceptance criteria
   ⚠ Created review issue mala-abc, epic blocked pending human decision
   ```

### Remediation Issue Format

When creating issues for unmet criteria:

**Title format**: `[Remediation] <criterion summary (first 60 chars)>`

**Description format**:
```markdown
## Context
This issue was auto-created by epic verification for epic `<epic_id>`.

## Unmet Criterion
<full criterion text>

## Evidence
<why this criterion was not met>

## Severity
<critical|major|minor>

## Resolution
Address this criterion to unblock epic closure. When complete, close this issue.
```

**Tags**: `epic_remediation:<epic_id>:<criterion_hash>`, `auto_generated`

**Priority**: Inherited from epic priority, adjusted by severity:
- critical: same as epic
- major: epic priority + 1 (lower priority)
- minor: epic priority + 2

**Deduplication**: Before creating, check for existing open issue with matching `epic_remediation:<epic_id>:<criterion_hash>` tag. If found, skip creation and reuse existing issue ID.

### Concurrency Handling

**Scenario**: Multiple epics become eligible simultaneously, or verification runs while another agent closes a child issue.

**Strategy**: Sequential processing with locking:

1. **Epic-level lock**: Before verifying an epic, acquire lock `epic_verify:<epic_id>`
2. **Processing order**: Process eligible epics sequentially (no parallel verification)
3. **Lock implementation**: Reuse existing `LockManager` from orchestrator
4. **Lock timeout**: 5 minutes (verification should complete within this)

**Rationale**: Parallel verification adds complexity (potential resource contention, duplicate remediation issues) with minimal benefit since epic closures are infrequent events.

```python
async def verify_and_close_eligible(self, ...) -> EpicVerificationResult:
    eligible_epics = await self._get_eligible_epics()
    results = {}

    for epic_id in eligible_epics:
        async with self.lock_manager.acquire(f"epic_verify:{epic_id}", timeout=300):
            verdict = await self.verify_epic(epic_id)
            results[epic_id] = verdict
            # ... handle verdict

    return EpicVerificationResult(...)
```

## User Experience

### Primary Flow

1. Developer completes final issue under an epic
2. Orchestrator closes the issue
3. Epic verifier detects epic is now eligible for closure
4. Verifier gathers child issue commits and generates scoped diff
5. Model analyzes diff against epic criteria
6. **If passed**: Epic closes automatically, log indicates verification passed
7. **If failed**:
   - Remediation issues created (one per unmet criterion, deduplicated)
   - Issues added as epic blockers
   - Epic remains open
   - Log shows which criteria weren't met

### Log Output Examples

**Success**:
```
◐ Verifying epic mala-xyz before closure...
◐ Collected 12 commits from 4 child issues (diff: 45KB)
◐ Epic mala-xyz: 5/5 acceptance criteria met (confidence: 0.92)
◐ Closed epic mala-xyz
```

**Failure**:
```
◐ Verifying epic mala-xyz before closure...
◐ Collected 8 commits from 3 child issues (diff: 32KB)
◐ Epic mala-xyz: 3/5 acceptance criteria met (confidence: 0.85)
◐ Unmet: "API endpoints must return proper error codes" (critical)
◐ Unmet: "All endpoints must be documented" (major)
◐ Created remediation issues: mala-abc, mala-def
◐ Epic mala-xyz blocked pending remediation
```

**Human Review Required**:
```
⚠ Epic mala-xyz flagged for human review: missing acceptance criteria
⚠ Created review issue mala-ghi, epic blocked pending human decision
⚠ Use --epic-override mala-xyz to force closure
```

**Large Diff**:
```
◐ Verifying epic mala-xyz before closure...
◐ Diff exceeds 100KB limit (actual: 256KB), using file-summary mode
◐ Epic mala-xyz: 4/5 acceptance criteria met (confidence: 0.78)
...
```

### Error States

- **Epic has no acceptance criteria**: Create human review issue, block closure
- **Cannot compute diff** (no child commits found): Create human review issue, block closure
- **Spec file referenced but not found**: Proceed with acceptance criteria only, log warning
- **Model verification timeout**: Retry per config, then create human review issue
- **Model returns low confidence (<0.5)**: Create human review issue, block closure

### Edge Cases

- **Missing acceptance criteria**:
  - Create human review issue with reason "No acceptance criteria defined"
  - Epic blocked until human resolves review issue or uses override
  - Log: `⚠ Epic mala-xyz has no acceptance criteria, flagged for human review`

- **Epic with no children**:
  - Should not reach verification (Beads won't mark as eligible)
  - If somehow triggered, create human review issue

- **Partial child completion** (some children failed/skipped):
  - Verification runs on available commits
  - Verdict reasoning notes incomplete context
  - May result in lower confidence, triggering human review

- **Duplicate verification run** (race condition):
  - Lock prevents concurrent verification of same epic
  - Second caller waits for lock, then sees updated state

## Implementation Plan

1. [ ] Add `EpicVerdict`, `UnmetCriterion`, `EpicVerificationResult` to `src/models.py`
2. [ ] Add `EpicVerificationModel` protocol to `src/protocols.py`
3. [ ] Add `RetryConfig` dataclass to `src/models.py`
4. [ ] Implement `ClaudeEpicVerificationModel` in `src/epic_verifier.py`
5. [ ] Implement `EpicVerifier` class with:
   - `verify_epic()` method
   - `verify_and_close_eligible()` method
   - Scoped diff computation (child commits only)
   - Large diff handling (tiered truncation)
   - Spec file parsing with expanded regex
6. [ ] Implement `create_remediation_issues()` with deduplication
7. [ ] Implement `add_epic_blockers()` using `bd dep add`
8. [ ] Implement `request_human_review()` for override flow
9. [ ] Add `--epic-override` CLI flag to orchestrator
10. [ ] Modify `orchestrator.py` to use `EpicVerifier` instead of direct `close_eligible_epics_async()`
11. [ ] Add locking for sequential epic processing
12. [ ] Add event sink integration for verification events
13. [ ] Write unit tests (see Testing Strategy)
14. [ ] Write integration tests (see Testing Strategy)

## Testing Strategy

### Unit Tests
- `test_scoped_diff_computation`: Verify only child commits included
- `test_scoped_diff_squashed_commits`: Handle squashed commit messages
- `test_scoped_diff_no_commits`: Return empty, trigger human review
- `test_large_diff_truncation`: Verify tiered truncation behavior
- `test_spec_path_extraction_nested`: Test nested directory paths
- `test_spec_path_extraction_case_insensitive`: Test .MD extension
- `test_verdict_parsing`: Ensure model output is correctly structured
- `test_remediation_issue_deduplication`: Verify existing issues are reused
- `test_remediation_issue_format`: Verify title/description/tags
- `test_human_review_creation`: Verify review issue format
- `test_missing_acceptance_criteria`: Confirm human review triggered
- `test_low_confidence_triggers_review`: Verify <0.5 threshold

### Integration Tests
- `test_epic_verification_pass`: Full flow with passing verification
- `test_epic_verification_fail_creates_issues`: Verify remediation issue creation
- `test_epic_with_spec_file`: Verification includes spec content
- `test_human_override_bypasses_verification`: Test --epic-override flag
- `test_model_timeout_creates_review`: Retry exhaustion behavior
- `test_concurrent_verification_locked`: Second caller waits for lock
- `test_dedup_across_retry`: Same criterion doesn't create duplicate issues

### Manual Testing
- Create epic with clear acceptance criteria
- Complete child issues that partially meet criteria
- Verify remediation issues are created correctly
- Verify epic remains open until all criteria met
- Test human override flow with missing criteria
- Test with large epic (>100KB diff) to verify truncation

## Decisions Made

- **Scoped diff, not baseline diff**: Only include commits linked to child issues via `bd-<child_id>:` prefix — prevents false pass/fail from unrelated work
- **Verification model is pluggable**: Claude initially, but protocol allows any model — supports future flexibility without code changes
- **One issue per gap, deduplicated**: Enables parallel remediation; criterion hash prevents duplicate issues across re-verification
- **Blockers via `bd dep add`**: Uses existing Beads dependency system rather than custom blocking mechanism
- **Not in BeadsClient**: Keeps BeadsClient focused on CLI wrapping; verification is orchestration-level logic
- **Structured verdict only**: No detailed markdown reports; keep output minimal and automation-friendly
- **Missing criteria = human review**: Consistent with goal of blocking closure; requires explicit override
- **Types in models.py**: Avoids circular import between protocols.py and epic_verifier.py
- **Sequential epic processing**: Simplicity over parallelism; epic closures are infrequent
- **Tiered large diff handling**: 100KB limit with file-summary fallback, then file-list mode
- **Human override via CLI flag and review issues**: Two paths — operator flag for immediate override, or close review issue for async approval
