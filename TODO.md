# Roadmap to v1

After config changes:
* failure mode and retry max not needed when commands are null
* 01:26:28 [trigger] ◦ [session_end] queued: success_count=7

Non-session review
* add to mala init
* Spawn fixer
* Logging
* config in mala.yaml

* show full command: Validation command(s) failed: gleam

* match trigger log color with the issue id up front
03:53:30 [casg-g03.4] ✓ GATE passed
03:53:30 [casg-g03.4] ✓ VALIDATE
03:53:30 [trigger] → [trigger] session_end started: issue_id=casg-g03.4
03:53:30 [trigger] ○ [trigger] session_end skipped: issue_id=casg-g03.4, reason=not_configured

refine logs:
04:00:49 [trigger] ◦ [run_end] command_started: setup (index=0)
04:00:49 [trigger] ◦ [run_end] command_completed: setup passed (0.0s)
04:00:49 [trigger] ◦ [run_end] command_started: format (index=1)

* Publish to PyPi

# Later

New Features
* Use cerberus for epic verification
* Add fixer sessions and cerberus reviews to logs search
* Worktree mode: each epic runs in a worktree, signals it is done to process in main branch, which merges it in + resolves conflicts
* CLI flag to control which beads issue types are processed (currently just tasks)
* Issue retry in same run
* Clean up uncommitted changes after agent timeout / soft kill?
* Implementers can use author context to communicate to reviewer

* Use Amp/Codex in the main agent loop (waiting until they have hooks)
  * Replace with new Edit tools

* CLI command for run statistics - tokens used, tools calls, reviewer/validation pass rates, etc.

Tech Debt
* Use pydantic-settings, or some other library for config
* Separate module used by reviewer and epic verifier for smart ticket creation: good descriptions, dependency awareness, deduplication
* Run architecture reviews on submodules

Config: Make it actually make sense / be consistent
* top level validation block that has commands and triggers under it?
* add config for evidence check, separate validation from code review?
* Separate mala agent logs from system claude code with CLAUDE_CONFIG_DIR env var

# Ideas
* Inter-agent communication
* Separate prompt/loop for bug fixes? red-green TDD
* Explore subagent - instruct use in the implementer prompt
* Stricter TDD

* rebuild in gleam?
