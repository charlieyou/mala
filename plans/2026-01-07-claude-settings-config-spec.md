# Claude Settings Configuration for Mala

**Tier:** M
**Owner:** Engineering Team
**Target ship:** Next Release
**Links:** Related to mala validation infrastructure

## 1. Outcome & Scope

**Problem / context**

Users running mala validation need different Claude Code CLI settings than their interactive development sessions. For example, validation runs may need stricter timeouts, specific models, or reduced max_turns to control costs. Currently, there's no way to configure Claude Code settings specifically for mala validation—agents inherit whatever project or user settings exist, leading to inconsistent or suboptimal validation behavior across different machines and contexts.

The Claude Agent SDK supports three predefined setting sources: `"local"` (`.claude/settings.local.json`), `"project"` (`.claude/settings.json`), and `"user"` (`~/.claude/settings.json`). The SDK merges these with local > project > user precedence. Users need to:
1. Define mala-specific Claude settings in `.claude/settings.local.json` (using the SDK's "local" source)
2. Configure which settings sources to merge (local/project/user) via `claude_settings_sources` in `mala.yaml` or `--claude-settings-sources` CLI flag
3. Optionally commit `.claude/settings.local.json` for reproducible validation environments (though it's typically gitignored for interactive use)

**Goal**

Enable users to configure Claude Code settings specifically for mala validation runs by controlling which settings sources the Claude Agent SDK uses, ensuring reproducible environments distinct from interactive user settings.

**Success criteria**
- Users can define a `.claude/settings.local.json` file for mala-specific validation settings
- Validation runs use the merged settings from configured sources (defaulting to `[local, project]`)
- Invalid settings cause a fast failure before any agents are spawned
- Settings are deterministic and reproducible when the same mala.yaml and settings files are used
- Users understand that `.claude/settings.local.json` should be committed for validation (unlike interactive use)

**Non-goals**
- Managing Claude Code API keys or authentication (handled by environment/existing mechanisms)
- Persistently modifying the global user configuration file
- Restricting which Claude Code settings can be used
- Modifying Claude Code CLI itself or its settings schema
- Supporting Anthropic API parameters directly (only Claude Code CLI settings)
- Per-agent or per-validation-rule settings (single merged config applies to all agents in a run)
- Hot-reloading settings during a validation run
- Implementing settings merge logic in mala (delegated to Claude Agent SDK)
- Validating Claude Code settings schema in mala (delegated to Claude Agent SDK)

## 2. User Experience & Flows

**Primary flow**
1. User creates `.claude/settings.local.json` in repository root with specific validation overrides (e.g., `{"timeout": 600}`)
2. User optionally adds `.claude/settings.local.json` to version control for reproducible validation
3. User optionally configures sources in `mala.yaml` (top-level key):
   ```yaml
   preset: python-uv
   claude_settings_sources: [local, project, user]  # top-level key
   commands:
     test: "uv run pytest"
   ```
   Or sets env var `MALA_CLAUDE_SETTINGS_SOURCES=local,project,user`
   Or passes CLI flag `--claude-settings-sources local,project,user`
4. User runs `mala run` or `mala epic-verify`
5. Mala resolves sources (CLI > Env Var > mala.yaml > default `[local, project]`)
6. Mala validates source names are in `{local, project, user}`
7. Mala passes the validated source names to the Claude Agent SDK via `setting_sources` parameter
8. The SDK loads and merges settings from the specified sources (local > project > user precedence)
9. Mala logs which sources were requested
10. The validation agent spawns using the SDK-merged settings

**Key states**
- **Valid Config**: Settings sources passed to SDK; agent starts and SDK handles merge
- **Invalid Config**: SDK reports schema validation failure; mala exits immediately with error
- **Missing 'local' File**: SDK silently skips missing file; mala optionally logs warning
- **Missing 'project'/'user' File**: SDK silently skips missing files

**Source file path mapping** (SDK-defined):
- `local` → `.claude/settings.local.json` (relative to repository root)
- `project` → `.claude/settings.json` (relative to repository root)
- `user` → `~/.claude/settings.json` (user's home directory)

## 3. Requirements + Verification

**R1 — Source Configuration**
- **Requirement:** The system MUST accept a list of settings sources via:
  - **mala.yaml** (top-level key `claude_settings_sources: [local, project, user]`)
  - **Environment variable** `MALA_CLAUDE_SETTINGS_SOURCES` (comma-separated: `local,project,user`)
  - **CLI argument** `--claude-settings-sources local,project,user` (comma-separated)

  Precedence (highest to lowest): CLI > Environment Variable > mala.yaml > default `[local, project]`.

  Valid sources are SDK-defined: `local`, `project`, `user`. Mala MUST validate source names and reject invalid values before passing to SDK. Mala MUST pass valid sources to the Claude Agent SDK via the `setting_sources` parameter in `ClaudeAgentOptions` (as documented in [Agent SDK reference](https://docs.anthropic.com/en/agent-sdk)).

  This configuration applies to ALL mala commands that spawn agents (`mala run`, `mala epic-verify`).

- **Verification:**
  - Given `mala.yaml` has top-level `claude_settings_sources: [local]`, when running `mala run`, then mala passes `setting_sources=["local"]` to SDK
  - Given env var `MALA_CLAUDE_SETTINGS_SOURCES=user,project`, when running `mala run` (no CLI flag, no mala.yaml), then mala passes `setting_sources=["user", "project"]` to SDK
  - Given `--claude-settings-sources local,project,user` CLI flag, when running `mala epic-verify`, then mala passes `setting_sources=["local", "project", "user"]` to SDK (CLI overrides env/yaml)
  - Given no config anywhere, when running any agent command, then mala passes `setting_sources=["local", "project"]` to SDK
  - Given invalid source `--claude-settings-sources foo,local`, when running, then mala exits with error "Invalid settings source 'foo'. Valid sources: local, project, user"

**R2 — SDK Settings Merge Assumption**
- **SDK Behavior Assumption:** The Claude Agent SDK merges settings from provided sources with local > project > user precedence (as documented). Mala relies on this SDK behavior and does not perform merging itself.
- **Mala's Responsibility:** Pass the correct source list to SDK; verify sources are passed correctly in unit tests.
- **Integration Testing:** E2E tests should verify that agent behavior reflects expected settings (e.g., timeout values are honored), confirming SDK merge works as expected.
- **Verification (mala unit tests):**
  - Given `claude_settings_sources: [local, project]` in mala.yaml, when mala constructs ClaudeAgentOptions, then `setting_sources=["local", "project"]` is passed to SDK
- **Verification (integration/E2E tests):**
  - Given `.claude/settings.local.json` with `{"timeout": 300}` and sources `[local, project]`, when agent runs, then agent session respects 300s timeout (verifies SDK merge behavior)

**R3 — Missing File Handling**
- **Requirement:** Mala SHOULD log an INFO message if `.claude/settings.local.json` does not exist when "local" is in configured sources, to help users verify their settings configuration. Mala MUST pass all configured sources to the SDK regardless of file existence; the SDK silently skips missing files.
- **Verification:**
  - Given `[local]` source configured and `.claude/settings.local.json` does not exist, when running, then mala logs INFO "Claude settings file .claude/settings.local.json not found (will be skipped)" and passes `setting_sources=["local"]` to SDK
  - Given `[project]` source configured and `.claude/settings.json` does not exist, when running, then no message is logged and mala passes `setting_sources=["project"]` to SDK
  - Given sources `[local, project]` with both files missing, when running, then log INFO about local only, pass both sources to SDK

**R4 — Schema Validation**
- **Requirement:** The Claude Agent SDK validates settings against its schema during initialization. If invalid, the SDK fails. Mala MUST catch SDK initialization errors and exit with a non-zero status code and descriptive error message. Mala does not perform schema validation itself.
- **Verification:**
  - Given `.claude/settings.local.json` contains `{"invalid_field": 123}` and mala passes `setting_sources=["local"]`, when SDK attempts initialization, then SDK fails with validation error and mala exits with "Claude Agent SDK error: [SDK error message]"
  - Given `.claude/settings.local.json` with malformed JSON, when SDK attempts to load, then SDK fails with parse error and mala exits with "Claude Agent SDK error: [SDK error message]"

**R5 — Startup Logging**
- **Requirement:** Mala MUST log the configured settings sources at INFO level before passing to the SDK. Mala MAY include file existence status in the log. Since the SDK performs the merge, mala cannot log individual setting values or their source attribution.
- **Verification:**
  - Given `setting_sources=["local", "project"]`, when validation starts, then mala logs "Claude settings sources: local, project"
  - Given `setting_sources=["user"]`, when validation starts, then mala logs "Claude settings sources: user"
  - Given "local" in sources and file missing, when validation starts, then mala logs INFO about missing file per R3

## 4. Instrumentation & Release Checks

**Instrumentation**
- Log the configured settings sources at INFO level during startup
- Track events:
  - Settings sources configured (which sources, via mala.yaml or CLI)
  - INFO log when `.claude/settings.local.json` is missing but requested
  - SDK initialization success/failure (captures validation errors)

**Decisions made**
- **Use SDK's "local" source**: After SDK research, determined the SDK only supports predefined sources ("local", "project", "user"), not custom names or paths. Use `.claude/settings.local.json` (SDK's "local" source) for mala-specific settings instead of inventing a custom "mala" source.
- **Commit .claude/settings.local.json for validation**: Though typically gitignored for interactive use, recommend committing this file for reproducible validation environments across CI/development.
- **Default `claude_settings_sources` to `[local, project]`**: Change from current SDK adapter default of `['project', 'user']` to `['local', 'project']`. Rationale: Prioritize validation-specific settings (local) over user settings, ensuring reproducible validation environments across CI and developer machines. This new default applies to ALL agent-based mala commands (`mala run`, `mala epic-verify`).
  - **Migration note**: Existing users without `.claude/settings.local.json` will see no behavior change (SDK skips missing local file). Users with existing `.claude/settings.local.json` will have it applied to validation runs. To maintain old behavior, explicitly set `claude_settings_sources: [project, user]` in mala.yaml.
- **Configuration architecture**: Parse `claude_settings_sources` from `mala.yaml` into `ValidationConfig` (in `src/domain/validation/config.py`), following existing pattern for mala.yaml fields. Thread this value through to `MalaConfig`/`ResolvedConfig` for use by `SDKClientFactory` in infra layer. Environment variable and CLI overrides handled in `MalaConfig.from_env()` consistent with other runtime config.
- **Top-level mala.yaml key**: Add `claude_settings_sources` as a top-level key in `mala.yaml` schema (alongside `preset`, `commands`, `coverage`, etc.) since it's a global SDK configuration.
- **Environment variable support**: Add `MALA_CLAUDE_SETTINGS_SOURCES` env var following existing mala config pattern (consistent with other global settings in `src/infra/io/config.py`).
- **Precedence order**: CLI > Env Var > mala.yaml > default, following standard configuration patterns.
- **Validate source names before SDK**: Catch invalid source names early with clear error messages rather than passing to SDK and getting cryptic failures.
- **Apply to all agent commands**: Use same settings sources for `mala run`, `mala epic-verify`, and any future agent-based commands for consistency.
- **Comma-separated CLI format**: `--claude-settings-sources local,project,user` is intuitive and common for list arguments.
- **Allow all Claude Code settings**: Trust users to configure appropriately; no restrictions on which settings can be used.
- **INFO log for missing local file**: Help users verify their configuration without being alarming (not a WARNING).
- **Delegate settings merge to SDK**: Mala passes source names to the Claude Agent SDK via `setting_sources` parameter; the SDK handles loading, merging (local > project > user), and validation. This avoids duplicating SDK logic and ensures consistency with Claude Code behavior.
- **No --dry-run mode for MVP**: R5 already logs configured sources at INFO level during startup. A dedicated --dry-run or --show-settings-sources flag adds minimal value and increases scope. Users can run with --verbose to see source configuration without executing validation. Defer this UX enhancement to post-MVP based on user feedback.

**Open questions**
- Should mala provide guidance in documentation about when to commit `.claude/settings.local.json` vs keeping it gitignored? (Recommendation: Document both patterns with clear use cases)
