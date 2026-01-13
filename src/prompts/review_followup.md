## External Review Failed (Attempt {attempt}/{max_attempts})

**Token Efficiency:** Use `read_range` ≤120 lines. No narration ("Let me..."). No git archaeology. No whole-file summaries. Fix directly.

The external reviewers found the following issues:

{review_issues}

**Required actions:**
1. Fix ALL issues marked as P0 or P1 above
2. Triage P2/P3 issues; fix the important ones, and explicitly note any you defer with rationale
3. **Use subagents whenever possible** for different files/parts of the codebase to save context
4. If needed, adjust the validation commands to scope them to only the files you touched
5. Re-run the full validation suite (same command set; may be scoped to touched files; commands are from project config):
```bash
{lint_command}
{format_command}
{typecheck_command}
{custom_commands_section}
{test_command}
```
6. Commit your changes with message: `bd-{issue_id}: <description>` (multiple commits allowed; use the prefix on each). Use `git add <files>` with explicit file paths only (no `-A`, `-u`, `--all`, directories, or globs) and commit in the same command.

**Disputing findings:** Your final summary message will be passed to the reviewer on retry. If you believe a finding is a **false positive**:
- State clearly: "P1 at line X is a false positive because..."
- Reference exact line numbers that establish invariants or guarantees
- Example: "Line 287 already calls `merge_config()` which guarantees non-None value"
- Be specific—vague defenses ("the code is correct") will be ignored

Note: The orchestrator will re-run both the quality gate and external review after your fixes.
