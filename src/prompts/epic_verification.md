# Epic Verification

You are verifying whether code changes satisfy an epic's acceptance criteria.

## Epic Acceptance Criteria

{epic_criteria}

## Spec Content (if available)

{spec_content}

## Scoped Diff (child issue commits only)

{diff_content}

## Task

Analyze the diff above and determine if ALL acceptance criteria have been met.

For each criterion:
1. Check if the implementation satisfies the requirement
2. Look for evidence in the code changes
3. Note any gaps or missing functionality

## Response Format

Respond with a JSON object containing:
- `passed`: boolean - true if ALL criteria are met
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

If all criteria are met:
```json
{
  "passed": true,
  "confidence": 0.92,
  "reasoning": "All acceptance criteria have been satisfied by the implementation.",
  "unmet_criteria": []
}
```
