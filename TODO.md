# Roadmap to v1

New Features
* Add code review to run-level validations
* Option to run run-level validations after every epic completion
* Rename gate 4 / run level validations
* Option to hard exit if run-level validations fail

Security
* Setup in devcontainer - should it just be compatible, or part of the app itself?

Compatibility
* Add claude reviewer

Config
* Add configs for which claude settings are used - user or project (or mala specific?)
* Interactive config setup
* Separate mala agent logs from system claude code with CLAUDE_CONFIG_DIR env var

* Publish to PyPi

# Later

New Features
* Use cerberus for epic verification
* Add fixer sessions and cerberus reviews to logs search
* Worktree mode: each epic runs in a worktree, signals it is done to process in main branch, which merges it in + resolves conflicts
* Add exit path for agents that are 
* CLI flag to control which beads issue types are processed (currently just tasks)
* Issue retry in same run
* Clean up uncommitted changes after agent timeout / soft kill

* Use Amp/Codex in the main agent loop (waiting until they have hooks)

* CLI command for run statistics - tokens used, tools calls, etc. 

Tech Debt
* Use pydantic-settings, or some other library for config
* Separate module used by reviewer and epic verifier for smart ticket creation: good descriptions, dependency awareness, deduplication

# Ideas
* Inter-agent communication
* Separate prompt/loop for bug fixes? red-green TDD
* Explore subagent - instruct use in the implementer prompt
