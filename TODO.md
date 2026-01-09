# Roadmap to v1

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
* refine --resume behavior -- should not affect prioritization 
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
* Run architecture reviews on submodules

* Generalize the quality gate commands into applying to different stages, eg. epic verification. Similar to what was done with the global validations

# Ideas
* Inter-agent communication
* Separate prompt/loop for bug fixes? red-green TDD
* Explore subagent - instruct use in the implementer prompt
