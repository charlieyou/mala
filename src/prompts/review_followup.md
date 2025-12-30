## External Review Failed (Attempt {attempt}/{max_attempts})

The external reviewers found the following issues:

{review_issues}

**Required actions:**
1. Fix ALL issues marked as ERROR above
2. Optionally address warnings if appropriate
3. Re-run the full validation suite:
   - `uv run pytest`
   - `uvx ruff check .`
   - `uvx ruff format .`
   - `uvx ty check`
4. Commit your changes with message: `bd-{issue_id}: <description>`

Note: The orchestrator will re-run both the quality gate and external review after your fixes.
