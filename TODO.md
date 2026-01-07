# Roadmap to v1

Bugs:
* --resume does not continue a session 
* fixers should be aware of the global validation commands so they can replicate runs

New Features
* Add code review to global kvalidations

Config
* Interactive config setup
* Separate mala agent logs from system claude code with CLAUDE_CONFIG_DIR env var
* cerberus args should be passed in via mala.yaml
* config reviewer/gate/etc retries in mala.yaml

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

* CLI command for run statistics - tokens used, tools calls, reviewer/validation pass rates, etc.

Tech Debt
* Use pydantic-settings, or some other library for config
* Separate module used by reviewer and epic verifier for smart ticket creation: good descriptions, dependency awareness, deduplication

# Ideas
* Inter-agent communication
* Separate prompt/loop for bug fixes? red-green TDD
* Explore subagent - instruct use in the implementer prompt
