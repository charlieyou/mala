## External Review Failed (Attempt {attempt}/{max_attempts})

**Token Efficiency:** Use `read_range` ≤120 lines. No narration ("Let me..."). No git archaeology. No whole-file summaries.

The external reviewers found the following issues:

{review_issues}

**Required actions:**
1. **Root-cause first**: For every P0/P1, state (max 2 sentences each):
   - **Why it's wrong**: Name the violated principle:
     - Invariant not enforced (assumed but not guaranteed)
     - Contract mismatch (caller/callee disagree on behavior)
     - Non-local break (change here breaks something elsewhere)
     - Spec misunderstanding (library/protocol doesn't work as assumed)
     - Missing validation (invalid input accepted)
     - Fail-open (error path allows unsafe operation)
   - **Why this fixes it**: The mechanism that enforces the invariant
   
   **Example:**
   > Reviewer: "INSERT...WHERE NOT EXISTS is not race-free"
   > Why wrong: Spec misunderstanding — PostgreSQL READ COMMITTED allows concurrent transactions to both see "not exists"
   > Why fix works: Unique partial index + ON CONFLICT makes the check-and-insert atomic
   
   If you can't name the principle clearly, return a `Blockers:` list instead of coding.

2. **Modeling Gate (if triggered)**: If any P0/P1 suggests hidden invariant, non-local effect, or subtle failure mode → write/update Operating Model (3-5 lines) before coding. See implementer prompt for format.

3. **Prove before fixing**: Add a regression test or assertion that fails before the fix and passes after. If impossible, state why and what surrogate evidence you'll use.

4. Fix ALL P0/P1 issues. Triage P2/P3; fix important ones, defer others with rationale.

5. Use subagents when >15 edits, >5 files, or multiple workstreams; otherwise stay single-agent.

6. Re-run validation suite:
```bash
{lint_command}
{format_command}
{typecheck_command}
{custom_commands_section}
{test_command}
```
7. Commit your changes with message: `bd-{issue_id}: <description>` (multiple commits allowed; use the prefix on each). Use `git add <files>` with explicit file paths only (no `-A`, `-u`, `--all`, directories, or globs) and commit in the same command.

**No pivot without proof:** Do not replace the approach (locks → checks → different primitive) unless you can show the previous approach fails via counterexample scenario or failing test. Prefer changing the enforcing mechanism over adding conditionals at call sites.

**Disputing findings:** Your final summary message will be passed to the reviewer on retry. If you believe a finding is a **false positive**:
- State clearly: "P1 at line X is a false positive because..."
- Reference exact line numbers that establish invariants or guarantees
- Example: "Line 287 already calls `merge_config()` which guarantees non-None value"
- Be specific—vague defenses ("the code is correct") will be ignored

**Output:** Use the same final output template as the implementer prompt (Implemented/Files changed/Tests/Quality checks/Commit/Lock contention/Follow-ups).

Note: The orchestrator will re-run both the quality gate and external review after your fixes.
