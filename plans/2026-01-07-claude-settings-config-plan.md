# Implementation Plan: Claude Settings Configuration

## Context & Goals

- **Spec**: `/home/cyou/mala/plans/2026-01-07-claude-settings-config-spec.md`
- Enable users to configure which Claude Code settings sources the SDK uses for mala validation runs
- Support configurable precedence (local/project/user) via mala.yaml, env var, or CLI flag
- Default to `[local, project]` for reproducible validation across CI and dev machines

## Scope & Non-Goals

### In Scope
- Add `claude_settings_sources` as top-level field in mala.yaml (parsed into ValidationConfig)
- Add `MALA_CLAUDE_SETTINGS_SOURCES` environment variable support
- Add `--claude-settings-sources` CLI flag for `mala run` and `mala epic-verify`
- Validate source names (must be `local`, `project`, or `user`) at config parse time
- Pass validated sources to Claude Agent SDK via `setting_sources` parameter
- Log resolved sources at INFO level during agent startup
- Log WARN message when `.claude/settings.local.json` is missing but "local" is in configured sources
- Change SDK adapter default from `['project', 'user']` to `['local', 'project']`
- Integration testing to verify settings from `.claude/settings.local.json` are respected by SDK

### Out of Scope (Non-Goals)
- Creating a custom "mala" settings source mechanism (we rely on SDK's "local" source)
- Modifying the Claude Agent SDK itself
- Implementing settings merge logic (delegated to Claude Agent SDK)
- Schema validation of Claude Code settings (delegated to SDK)
- Per-agent or per-validation-rule settings (single config applies to all agents)
- E2E testing for this specific feature (integration tests are sufficient)
- Hot-reloading settings during a run
- Managing API keys or authentication

## Assumptions & Constraints

### Implementation Constraints
- Claude Agent SDK supports passing a list of `setting_sources`
- `.claude/settings.local.json` is the standard path for local settings recognized by SDK (mapped to "local" source)
- SDK only supports predefined sources: `"local"`, `"project"`, `"user"`
- SDK handles settings merge with local > project > user precedence
- SDK performs schema validation and fails on invalid settings
- **Layering**: Strictly follow Domain → Infra → CLI layering (Domain defines schema, Infra handles environment/resolution, CLI handles user input)
- **Validation**: Fail fast at config parse time
- **Logging**: Log at runtime (agent spawn) for visibility

### Testing Constraints
- Unit tests must verify source name validation
- Unit tests must verify correct sources are passed to SDK
- Integration tests must verify agent behavior reflects expected settings (timeout values honored)
- Must maintain existing test coverage threshold (72%)

## Prerequisites

- [x] Access to `mala` codebase and `plans/` directory
- [x] Claude Agent SDK supports `setting_sources` parameter (confirmed in codebase)

## High-Level Approach

The implementation follows mala's existing configuration layering pattern (domain → infra → CLI). Configuration flows from mala.yaml (domain layer) through environment variables and CLI overrides (infra layer) to the SDK adapter (runtime layer). Source name validation happens at config parse time (ValidationConfig.from_dict, MalaConfig.from_env) to fail fast with clear error messages. Logging happens at agent spawn time in AgentRuntimeBuilder to show the final resolved sources passed to the SDK for each agent.

Implementation steps:
1. **Domain Configuration**: Update `ValidationConfig` (mala.yaml) to parse and validate `claude_settings_sources`
2. **Infra Configuration**: Update `MalaConfig` to accept `claude_settings_sources` parameter (for orchestrator to pass mala.yaml value) and parse `MALA_CLAUDE_SETTINGS_SOURCES` env var
3. **Orchestrator Integration**: Factory/orchestrator loads ValidationConfig.claude_settings_sources and passes to MalaConfig constructor, implementing precedence: Env Var > mala.yaml > default
4. **CLI Updates**: Add `--claude-settings-sources` flag to `run` and `epic-verify` commands, updating `CLIOverrides`
5. **Config Resolution**: Merge config layers in `build_resolved_config()`, applying final precedence: CLI > (Env Var > mala.yaml > default)
6. **Validation**: Validate source names early (at config parse time) for clear error messages listing valid sources
7. **Runtime & SDK**: Pass resolved sources to `AgentRuntimeBuilder`, perform file existence checks (logging WARN if missing), and inject into SDK client options
8. **Logging**: Log final resolved sources at INFO level in AgentRuntimeBuilder.build() before creating SDK options

## Technical Design

### Architecture

The architecture follows mala's existing layered pattern with clear boundaries:

**Layer 1 - Domain (ValidationConfig):**
- Parse `claude_settings_sources` from mala.yaml top-level key
- Validate source names are in `{'local', 'project', 'user'}`
- Store as `tuple[str, ...] | None` (None means "use default")
- Boundary: Exposes immutable config to infra layer

**Layer 2 - Infra (MalaConfig/ResolvedConfig):**
- MalaConfig.from_env() reads `MALA_CLAUDE_SETTINGS_SOURCES` env var (comma-separated)
- MalaConfig constructor accepts optional `claude_settings_sources` parameter for orchestrator to pass mala.yaml value
- MalaConfig stores default `('local', 'project')` if not configured
- CLIOverrides captures raw CLI flag value (comma-separated string)
- build_resolved_config() applies precedence: CLI > Env Var > mala.yaml > default
- ResolvedConfig holds final `claude_settings_sources: tuple[str, ...]`
- Boundary: Provides resolved config to orchestrator/runtime

**Orchestrator/Factory Integration (bridges Domain → Infra):**
- Factory/orchestrator loads ValidationConfig from mala.yaml
- Passes ValidationConfig.claude_settings_sources to MalaConfig constructor (if present)
- This implements the mala.yaml layer in the precedence chain before CLI resolution

**Layer 3 - CLI:**
- Parse `--claude-settings-sources` flag in run() and epic_verify()
- Pass raw string to CLIOverrides for parsing in build_resolved_config()
- Boundary: Translates user input to config overrides

**Layer 4 - Runtime (AgentRuntimeBuilder):**
- Accept `setting_sources: list[str]` in __init__()
- Log final sources at INFO level in build() before creating SDK options
- Check if `.claude/settings.local.json` exists when "local" in sources; log WARN if missing
- Pass `setting_sources` to sdk_client_factory.create_options()
- Boundary: Bridges config to SDK

**Layer 5 - SDK Adapter (SDKClientFactory):**
- Accept `setting_sources: list[str] | None` parameter in create_options()
- Default to `["local", "project"]` (changed from `["project", "user"]`)
- Pass to ClaudeAgentOptions constructor
- Boundary: Encapsulates SDK interaction

### Data Model

Configuration precedence chain:
```
CLI flag > Environment variable > mala.yaml > default ['local', 'project']
```

**Data flow through layers:**

1. **mala.yaml** → ValidationConfig.claude_settings_sources: tuple[str, ...] | None
   - Example: `claude_settings_sources: [local, project, user]` → `('local', 'project', 'user')`
   - None means "not configured in mala.yaml, use default"

2. **Env var** → MalaConfig.claude_settings_sources: tuple[str, ...]
   - Parse `MALA_CLAUDE_SETTINGS_SOURCES=local,project,user` → `('local', 'project', 'user')`
   - Filter empty parts: `local,,project` → `('local', 'project')`
   - Default: `('local', 'project')`

3. **CLI flag** → CLIOverrides.claude_settings_sources: str | None
   - Example: `--claude-settings-sources local,project,user` → `"local,project,user"`
   - Raw string, parsed in build_resolved_config()

4. **mala.yaml → MalaConfig integration** (orchestrator/factory layer):
   - The orchestrator loads ValidationConfig from mala.yaml
   - If ValidationConfig.claude_settings_sources is not None AND MalaConfig has default value:
     - Pass ValidationConfig.claude_settings_sources to MalaConfig constructor (overriding env var default)
   - This happens in the factory/orchestrator before calling build_resolved_config()
   - Precedence at this stage: Env var > mala.yaml > default

5. **Resolution** → ResolvedConfig.claude_settings_sources: tuple[str, ...]
   - Apply precedence in build_resolved_config():
     - If CLI override: parse and validate CLI string
     - Else: use base_config.claude_settings_sources (which already has Env > mala.yaml > default applied)
   - Final precedence chain: CLI > Env Var > mala.yaml > default

6. **Runtime** → AgentRuntimeBuilder(setting_sources=list[str])
   - Convert tuple to list: `('local', 'project')` → `['local', 'project']`
   - Pass to SDK adapter

7. **SDK** → ClaudeAgentOptions(setting_sources=['local', 'project'])
   - SDK merges settings from configured sources (local > project > user precedence)

### API/Interface Design

**ValidationConfig changes** (src/domain/validation/config.py):
- Add optional `claude_settings_sources: tuple[str, ...] | None` field
- Parse from top-level mala.yaml key (same level as `preset`, `commands`, `coverage`)
- Validate source names: each must be in `{'local', 'project', 'user'}`
- Error message format: "Invalid Claude settings source 'foo'. Valid sources: local, project, user"
- Empty list `[]` is valid (passed to SDK as-is)

**MalaConfig changes** (src/infra/io/config.py):
- Add `claude_settings_sources: tuple[str, ...]` field with default `('local', 'project')`
- Update constructor to accept optional `claude_settings_sources` parameter (for orchestrator to pass ValidationConfig value)
- Parse `MALA_CLAUDE_SETTINGS_SOURCES` env var in `from_env()` (comma-separated)
- Split by comma, filter empty parts (forgiving): `local,,project` → `('local', 'project')`
- Validate source names at parse time: reject with ConfigurationError if invalid
- Error format: "MALA_CLAUDE_SETTINGS_SOURCES: Invalid source 'foo'. Valid sources: local, project, user"
- **Orchestrator integration**: The factory/orchestrator will load ValidationConfig.claude_settings_sources and pass to MalaConfig constructor if present, implementing mala.yaml precedence over env var default

**CLIOverrides changes** (src/infra/io/config.py):
- Add `claude_settings_sources: str | None` field for raw CLI flag value
- Parsing happens in build_resolved_config() using helper function
- Split by comma, filter empty parts, validate source names
- Error format: "CLI: Invalid source 'foo'. Valid sources: local, project, user"

**ResolvedConfig changes** (src/infra/io/config.py):
- Add `claude_settings_sources: tuple[str, ...]` field
- Merge logic in build_resolved_config():
  1. If cli_overrides.claude_settings_sources is not None: parse, validate, use
  2. Else: use base_config.claude_settings_sources (from env var or default)

**CLI changes** (src/cli/cli.py):
- Add `--claude-settings-sources` option to both `run()` and `epic_verify()` commands
- Type: `str | None`, comma-separated value (e.g., "local,project,user")
- Help text: "Comma-separated list of Claude settings sources (local, project, user). Default: local,project"

**SDK Adapter changes** (src/infra/sdk_adapter.py):
- Change default `setting_sources` from `["project", "user"]` to `["local", "project"]`
- Accept `setting_sources: list[str] | None` parameter (no change needed, already exists)
- Default expression: `setting_sources or ["local", "project"]` (change from `["project", "user"]`)

**Agent Runtime changes** (src/infra/agent_runtime.py):
- Add `setting_sources: list[str] | None = None` parameter to AgentRuntimeBuilder.__init__()
- Store as `self._setting_sources`
- In build() method:
  1. Log final sources at INFO level: "Claude settings sources: local, project"
  2. Check if `.claude/settings.local.json` exists when "local" in sources; log WARN if missing:
     "Claude settings file .claude/settings.local.json not found (will be skipped)"
  3. Pass `setting_sources=self._setting_sources` to sdk_client_factory.create_options()

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/domain/validation/config.py` | Exists | Add `claude_settings_sources` field to `ValidationConfig` with validation logic |
| `src/infra/io/config.py` | Exists | Update `MalaConfig`, `CLIOverrides`, `ResolvedConfig` with env var parsing and precedence resolution |
| `src/cli/cli.py` | Exists | Add `--claude-settings-sources` flag to `run` and `epic_verify` commands |
| `src/infra/agent_runtime.py` | Exists | Update builder to accept sources, check file existence, log resolved sources |
| `src/infra/sdk_adapter.py` | Exists | Change default sources from `["project", "user"]` to `["local", "project"]` |
| `docs/project-config.md` | Exists | Document new `claude_settings_sources` field in mala.yaml |
| `tests/unit/domain/test_config.py` | Exists | Test `mala.yaml` parsing and validation |
| `tests/unit/infra/test_config.py` | Exists | Test env var parsing, precedence, and defaults |
| `tests/integration/pipeline/test_agent_session_runner.py` | Exists | Test SDK respects provided settings (timeout values honored) |

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes
- **Invalid Source**: User provides "foo"
  - *Handling*: `ConfigError` at parse time with message "Invalid Claude settings source 'foo'. Valid sources: local, project, user"
- **Missing Local File**: User requests "local" but `.claude/settings.local.json` doesn't exist
  - *Handling*: Log `WARN` in `AgentRuntimeBuilder`: "Claude settings file .claude/settings.local.json not found (will be skipped)", proceed (SDK handles missing files gracefully)
- **Empty List**: User provides empty list `[]`
  - *Handling*: Allowed. Pass to SDK; SDK uses defaults or no settings
- **Malformed env var**: `MALA_CLAUDE_SETTINGS_SOURCES=local,,project` (double commas)
  - *Handling*: Split by comma, filter empty parts (forgiving parsing)
- **Malformed JSON**: Invalid JSON in `.claude/settings.local.json`
  - *Handling*: SDK fails with parse error, mala exits with descriptive message
- **Mixed valid/invalid sources**: User provides `[local, foo, project]`
  - *Handling*: Fail on first invalid source with clear error message

### Breaking Changes & Compatibility
- **Default Behavior Change**: Default sources change from `['project', 'user']` (SDK default) to `['local', 'project']`
  - *Impact*: Users relying on global user settings for validation will need to explicitly opt-in or migrate settings to `mala.yaml` or `.claude/settings.local.json`
  - *Mitigation*: Documented change in release notes. This is intentional to improve reproducibility
  - *Migration Path*: To restore old behavior, explicitly set `claude_settings_sources: [project, user]` in mala.yaml
- **No schema version bump needed**: This is a backward-compatible addition (new optional field)

## Testing & Validation Strategy

### Unit Tests

**ValidationConfig tests** (tests/unit/domain/test_config.py or similar):
- Parse valid sources from mala.yaml: `claude_settings_sources: [local, project]` → `('local', 'project')`
- Reject invalid source name: `claude_settings_sources: [foo]` → ConfigError with "Invalid Claude settings source 'foo'. Valid sources: local, project, user"
- Accept empty list: `claude_settings_sources: []` → `tuple()`
- Default to None when field omitted

**MalaConfig tests** (tests/unit/infra/test_config.py):
- Parse env var: `MALA_CLAUDE_SETTINGS_SOURCES=local,project` → `('local', 'project')`
- Filter empty parts: `MALA_CLAUDE_SETTINGS_SOURCES=local,,project` → `('local', 'project')`
- Reject invalid source in env var: `MALA_CLAUDE_SETTINGS_SOURCES=foo` → ConfigurationError
- Default to `('local', 'project')` when env var not set

**CLIOverrides + ResolvedConfig tests** (tests/unit/infra/test_config.py):
- CLI override precedence: CLI flag overrides env var and mala.yaml
- Env var precedence: env var overrides mala.yaml but not CLI
- Default when no override: use `('local', 'project')`
- Parse CLI value: `"local,project,user"` → `('local', 'project', 'user')`
- Reject invalid source in CLI: raises ValueError with "CLI: Invalid source 'foo'. Valid sources: local, project, user"

**Coverage requirement:** Maintain existing test coverage threshold (72%)

### Integration Tests
- **SDK Settings Merge Test** (tests/integration/pipeline/test_agent_session_runner.py):
  - Create a temporary `.claude/settings.local.json` with a distinct `timeout` value
  - Configure `mala` to use `['local']` sources
  - Run a lightweight agent task/verification
  - Assert that the SDK client was initialized with the timeout from the JSON file (confirms SDK respects settings merge)
- **Missing File Handling Test**:
  - Verify WARN log appears when `.claude/settings.local.json` missing and "local" in sources

### E2E Tests
- Not required per user decision

### Manual Verification
- Run `mala run` with `--claude-settings-sources=local,user` and verify log output shows the correct sources
- Test invalid source name to verify error message quality
- Test missing `.claude/settings.local.json` to verify WARN log appears
- Verify precedence: CLI > Env > YAML > Default

### Acceptance Criteria Coverage

| Spec AC | Covered By |
|---------|------------|
| R1: Configure sources via file/env/CLI | Unit tests for `config.py` (Domain & Infra), CLI flag implementation |
| R2: Pass sources to SDK (merge delegation) | Integration test `test_agent_session_runner.py` verifying SDK respects settings |
| R3: Handle missing setting files | Unit test for `AgentRuntimeBuilder` logging (WARN when file missing) |
| R4: Validate source names | Unit tests for `ValidationConfig` and `MalaConfig` (reject invalid sources at parse time) |
| R5: Log active sources at startup | Unit test for `AgentRuntimeBuilder` logging (INFO level) |

## Open Questions

None. All decisions have been finalized through the spec and interview process.

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
