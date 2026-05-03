## Global Validation Failed (Attempt {attempt}/{max_attempts})

Continue following the implementer prompt where applicable. This prompt overrides it only where explicitly stated.

**Failed command:** `{failed_command}`

The global validation found issues that need to be fixed:

{failure_output}

## Required Actions

This is a global validation phase after per-session work is complete. The full validation suite must pass, so fix all errors required by the failed command, even if they appear in files that were not touched by any prior agent.

1. Analyze the validation failure output above.
2. Fix the root cause rather than suppressing the error.
3. Re-run the full validation suite on the entire codebase:

{validation_commands}

4. If formatting changes files, re-run validations from the start.
5. Commit fixes with:

```bash
git add <explicit files> && git commit -m "bd-run-validation: <description>"
```

Use explicit file paths only. Do not use `-A`, `-u`, `--all`, directories, globs, or `git commit -a`.

Do not release any locks; the orchestrator handles cleanup for this phase.
