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

* deadlock victims should have their changes restored?

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
