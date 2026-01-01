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

## Response Format

Respond with a JSON object containing:
- `passed`: boolean - true if ALL code-related criteria are met
- `confidence`: float (0.0-1.0) - confidence in your assessment
- `reasoning`: string - explanation of your verification outcome
- `unmet_criteria`: array of objects for any unmet criteria, each with:
  - `criterion`: the specific requirement not met
  - `evidence`: why it's considered unmet
  - `severity`: "critical" | "major" | "minor"

Example response:
```json
{
  "passed": false,
  "confidence": 0.85,
  "reasoning": "Most acceptance criteria are met, but error handling is incomplete.",
  "unmet_criteria": [
    {
      "criterion": "API endpoints must return proper error codes",
      "evidence": "The /users endpoint returns 500 for invalid input instead of 400",
      "severity": "major"
    }
  ]
}
```

If all code-related criteria are met:
```json
{
  "passed": true,
  "confidence": 0.92,
  "reasoning": "All code-related acceptance criteria have been satisfied by the implementation.",
  "unmet_criteria": []
}
```
