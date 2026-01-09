# Roadmap to v1

Bugs
* Ruff is making fixes in global val that are not being committed
* failure mode and retry max not needed when commands are null

New Features
* Add code review to global validations

Config
* Interactive config setup
* Separate mala agent logs from system claude code with CLAUDE_CONFIG_DIR env var
* cerberus args should be passed in via mala.yaml
* config reviewer/gate/etc retries in mala.yaml

* Publish to PyPi

# Later

Bugs
* Locks are not cleaned up after interrupted tests

New Features
* Use cerberus for epic verification
* Add fixer sessions and cerberus reviews to logs search
* Worktree mode: each epic runs in a worktree, signals it is done to process in main branch, which merges it in + resolves conflicts
* Add exit path for agents that are 
* CLI flag to control which beads issue types are processed (currently just tasks)
* Issue retry in same run
* Clean up uncommitted changes after agent timeout / soft kill
* Implementers can use author context to communicate to reviewer

* Use Amp/Codex in the main agent loop (waiting until they have hooks)

* CLI command for run statistics - tokens used, tools calls, reviewer/validation pass rates, etc.

Tech Debt
* Use pydantic-settings, or some other library for config
* Separate module used by reviewer and epic verifier for smart ticket creation: good descriptions, dependency awareness, deduplication
* Run architecture reviews on submodules

Config: Make it actually make sense / be consistent
* top level validation block that has commands and triggers under it?
* add config for evidence check, separate validation from code review?

# Ideas
* Inter-agent communication
* Separate prompt/loop for bug fixes? red-green TDD
* Explore subagent - instruct use in the implementer prompt
