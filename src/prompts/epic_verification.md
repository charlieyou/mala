# Epic Verification

You are verifying whether code changes satisfy an epic's acceptance criteria.

You must explore the repository yourself using the commit scope below. Do not rely on a provided diff.
Non-code criteria (tests, linting, formatting, coverage, CI/deploy, docs) are out of scope and must NOT be listed as unmet criteria.

## Epic Acceptance Criteria

{epic_criteria}

## Spec Content (if available)

{spec_content}

## Commit Scope (child issue commits only)

Commit range hint:
{commit_range}

Commit list (authoritative):
{commit_list}

## Task

Analyze the code changes from the commits above and determine if ALL code-related acceptance criteria have been met.

For each code-related criterion:
1. Check if the implementation satisfies the requirement
2. Look for evidence in the repository/commits
3. Note any gaps or missing functionality

If an acceptance criterion is purely non-code (tests/lints/format/coverage/CI/docs/deploy), ignore it and do not include it in `unmet_criteria`.

## Priority Levels (matching Cerberus)

- **P0**: Drop everything. Blocking release or major usage. Critical functional gaps.
- **P1**: Urgent. Should address in next cycle. Significant behavioral issues.
- **P2**: Normal. Fix eventually. Minor gaps, edge cases, polish items. **Non-blocking.**
- **P3**: Low. Nice to have. Style preferences, subjective improvements. **Non-blocking.**

**Blocking vs Non-blocking:**
- P0/P1 issues block epic closure and create remediation tasks
- P2/P3 issues are informational only - they are noted but do NOT block epic closure

**Style/Subjective Criteria:**
Code-style constraints (function length limits, naming preferences, refactoring suggestions) should be P3 when the implementation is functionally correct. Example: "MalaOrchestrator.__init__ under 60 lines" is P3 if the factory function works correctly but the method is 84 lines.

## Response Format

Respond with a JSON object containing:
- `passed`: boolean - true if ALL P0/P1 criteria are met (P2/P3 issues don't block)
- `confidence`: float (0.0-1.0) - confidence in your assessment
- `reasoning`: string - explanation of your verification outcome
- `unmet_criteria`: array of objects for any unmet criteria, each with:
  - `criterion`: the specific requirement not met
  - `evidence`: why it's considered unmet
  - `priority`: 0 | 1 | 2 | 3 (matching Cerberus priority levels)

Example response with blocking issue:
```json
{
  "passed": false,
  "confidence": 0.85,
  "reasoning": "Most acceptance criteria are met, but error handling is incomplete.",
  "unmet_criteria": [
    {
      "criterion": "API endpoints must return proper error codes",
      "evidence": "The /users endpoint returns 500 for invalid input instead of 400",
      "priority": 1
    }
  ]
}
```

Example with only non-blocking (P2/P3) issues:
```json
{
  "passed": true,
  "confidence": 0.88,
  "reasoning": "All functional criteria met. Only a style preference remains as P3.",
  "unmet_criteria": [
    {
      "criterion": "MalaOrchestrator.__init__ under 60 lines",
      "evidence": "MalaOrchestrator.__init__ is 84 lines, but factory function works correctly.",
      "priority": 3
    }
  ]
}
```

If all criteria are met:
```json
{
  "passed": true,
  "confidence": 0.92,
  "reasoning": "All code-related acceptance criteria have been satisfied by the implementation.",
  "unmet_criteria": []
}
```
