# Implementation Plan: Language-Agnostic Configuration via mala.yaml

## Context & Goals
- **Spec**: `docs/2026-01-01-language-agnostic-config-spec.md`
- Replace hard-coded Python commands with user-configurable `mala.yaml`
- Provide built-in presets for common stacks (python-uv, node-npm, go, rust)
- Support preset extension with user overrides
- Make coverage reporting optional and configurable (Cobertura XML for MVP)
- Support user-defined file patterns for code change detection

## Scope & Non-Goals

### In Scope
- Configuration file (`mala.yaml`) loading and validation
- Preset registry and merge logic
- 6 command types: setup, test, lint, format, typecheck, e2e
- Code pattern matching for change detection
- Tool name extraction from command strings
- Coverage threshold enforcement (Cobertura XML only)
- Error handling with clear, specific messages

### Out of Scope (Non-Goals)
- Auto-detection of project type without explicit configuration
- Backwards compatibility fallback to Python defaults
- Migration tooling or `mala init` command
- Monorepo support with multiple languages
- JSON and LCOV coverage formats (deferred to follow-up)
- Custom command types beyond the six built-in kinds
- Remote/dynamic preset fetching

## Assumptions & Constraints

### Implementation Constraints
- [TBD: Should configuration be loaded lazily or eagerly at startup?]
- [TBD: Shell execution (shell=True) vs list execution - confirm spec decision]
- Python 3.10+ compatibility required (from existing codebase)
- Must work with existing CommandRunner infrastructure

### Testing Constraints
- Coverage threshold: 85% (from CLAUDE.md)
- Unit tests for all new modules
- Integration tests for config loading + validation flow
- [TBD: How to test preset loading from package data?]

## Prerequisites
- [ ] [TBD: Any new dependencies needed? pyyaml is likely already available]
- [ ] Review existing ValidationSpec to ensure no breaking changes to consumers

## High-Level Approach
[TBD: Detailed approach after interview - generally:
1. Create config schema and loader
2. Create preset registry
3. Implement config merger
4. Replace build_validation_spec() with config-driven builder
5. Update coverage/baseline services
6. Update lint cache patterns
7. Update quality gate messaging]

## Detailed Plan

### Task 1: Create Configuration Schema and Dataclass
- **Goal**: Define ValidationConfig dataclass matching mala.yaml schema
- **Covers**: AC "Given a mala.yaml with invalid syntax... fails fast with specific error"
- **Depends on**: None
- **Changes**:
  - New: `src/domain/validation/config.py` — ValidationConfig dataclass, schema types
  - New: `tests/test_validation_config.py` — Unit tests for config dataclass
- **Verification**: [TBD: How to test schema validation?]
- **Rollback**: Delete new files

### Task 2: Implement YAML Config Loader
- **Goal**: Load and validate mala.yaml from repo root
- **Covers**: AC "Given a project with no mala.yaml... fails fast with clear error"
- **Depends on**: Task 1
- **Changes**:
  - New: `src/domain/validation/config_loader.py` — load_config() function
  - Modify: `src/domain/validation/spec.py` — Call config loader from build_validation_spec
  - New: `tests/test_config_loader.py` — Unit tests
- **Verification**: [TBD]
- **Rollback**: Revert spec.py changes, delete new files

### Task 3: Create Preset Registry
- **Goal**: Discover and load built-in presets from package data
- **Covers**: AC "Given a mala.yaml that specifies preset: python-uv..."
- **Depends on**: Task 1
- **Changes**:
  - New: `src/domain/validation/preset_registry.py` — PresetRegistry class
  - New: `src/domain/validation/presets/` — Directory for preset YAML files
  - New: `src/domain/validation/presets/python-uv.yaml`
  - New: `src/domain/validation/presets/node-npm.yaml`
  - New: `src/domain/validation/presets/go.yaml`
  - New: `src/domain/validation/presets/rust.yaml`
  - Modify: `pyproject.toml` — Include presets as package data
  - New: `tests/test_preset_registry.py` — Unit tests
- **Verification**: [TBD: Test that presets load correctly from installed package]
- **Rollback**: Delete new directory and files

### Task 4: Implement Config Merger
- **Goal**: Merge preset config with user overrides following spec rules
- **Covers**: AC "preset with custom test command override uses preset except for override"
- **Depends on**: Task 1, Task 3
- **Changes**:
  - New: `src/domain/validation/config_merger.py` — merge_configs() function
  - New: `tests/test_config_merger.py` — Unit tests for all merge rules
- **Verification**: [TBD]
- **Rollback**: Delete new files

### Task 5: Implement Tool Name Extractor
- **Goal**: Extract tool names from command strings for logging/evidence
- **Covers**: AC "Given a command string 'npx eslint .', extracts 'eslint'"
- **Depends on**: None
- **Changes**:
  - New: `src/domain/validation/tool_name_extractor.py` — extract_tool_name() function
  - New: `tests/test_tool_name_extractor.py` — Unit tests with examples from spec
- **Verification**: Test all examples from spec document
- **Rollback**: Delete new files

### Task 6: Implement Code Pattern Matcher
- **Goal**: Match file paths against user-defined glob patterns
- **Covers**: AC "custom code_patterns... only files matching trigger validation"
- **Depends on**: None
- **Changes**:
  - New: `src/domain/validation/code_pattern_matcher.py` — glob_to_regex, matches_pattern functions
  - New: `tests/test_code_pattern_matcher.py` — Unit tests with examples from spec
- **Verification**: Test examples from spec: `*.py`, `src/**/*.py`, `**/test_*.py`
- **Rollback**: Delete new files

### Task 7: Create Config-Driven Spec Builder
- **Goal**: Replace hard-coded build_validation_spec with config-driven version
- **Covers**: Multiple ACs for Go, Node.js, optional commands
- **Depends on**: Tasks 1-6
- **Changes**:
  - Modify: `src/domain/validation/spec.py` — Rewrite build_validation_spec() to use config
  - Modify: `tests/test_spec.py` — Update tests for new behavior
- **Verification**: [TBD: Integration test with sample mala.yaml files?]
- **Rollback**: Revert spec.py changes

### Task 8: Update Coverage Parsing for Config
- **Goal**: Use configured coverage format, file, threshold
- **Covers**: AC "coverage enabled with format: xml... parses and enforces threshold"
- **Depends on**: Task 7
- **Changes**:
  - Modify: `src/domain/validation/coverage.py` — Support config-driven coverage settings
  - Modify: `tests/test_coverage.py` — Add tests for config-driven coverage
- **Verification**: [TBD]
- **Rollback**: Revert coverage.py changes

### Task 9: Generalize BaselineCoverageService
- **Goal**: Execute configured commands instead of hard-coded pytest
- **Covers**: Coverage baseline refresh with any test command
- **Depends on**: Task 7, Task 8
- **Changes**:
  - Modify: `src/domain/validation/coverage.py` — Generalize BaselineCoverageService
  - Modify: `tests/test_coverage.py` — Add tests for generalized baseline
- **Verification**: [TBD: Test with non-pytest test command?]
- **Rollback**: Revert changes

### Task 10: Update Lint Cache Detection Patterns
- **Goal**: Auto-derive detection patterns from configured commands
- **Covers**: Evidence detection for any lint/format/typecheck tool
- **Depends on**: Task 5, Task 7
- **Changes**:
  - Modify: `src/infra/hooks/lint_cache.py` — Replace hard-coded LINT_COMMAND_PATTERNS
  - Modify: `tests/test_lint_cache.py` — Update tests
- **Verification**: [TBD]
- **Rollback**: Revert changes

### Task 11: Update Quality Gate Messaging
- **Goal**: Derive tool names from spec instead of hard-coded KIND_TO_NAME
- **Covers**: Human-readable error messages for any tool
- **Depends on**: Task 5, Task 7
- **Changes**:
  - Modify: `src/domain/quality_gate.py` — Use spec's tool names
  - Modify: `tests/test_quality_gate.py` — Update tests
- **Verification**: [TBD]
- **Rollback**: Revert changes

### Task 12: Update RepoType Detection
- **Goal**: Replace Python-specific detection with mala.yaml presence check
- **Covers**: AC "no mala.yaml... fails fast with clear error"
- **Depends on**: Task 2
- **Changes**:
  - Modify: `src/domain/validation/spec.py` — Update detect_repo_type or remove
  - Modify: `tests/test_spec.py` — Update detection tests
- **Verification**: [TBD]
- **Rollback**: Revert changes

### Task 13: Integration Testing
- **Goal**: End-to-end tests for complete validation flows
- **Covers**: All acceptance criteria
- **Depends on**: Tasks 1-12
- **Changes**:
  - New: `tests/test_validation_config_integration.py` — Integration tests
  - New: `tests/fixtures/mala-configs/` — Sample mala.yaml files for testing
- **Verification**: All ACs pass with sample configs
- **Rollback**: Delete new files

## Risks, Edge Cases & Breaking Changes

### Risks
- [TBD: Risk of breaking existing Python projects during transition?]
- [TBD: Shell command security considerations?]
- [TBD: Performance impact of YAML parsing on every run?]

### Edge Cases (from spec)
- Empty commands section with preset → Use all preset commands
- Command set to null → Explicitly disable even if preset defines it
- No commands at all → Config validation error
- All commands explicitly set to null → Config validation error
- Command set to empty string → Config validation error
- Empty code_patterns list → Treat all files as code
- Coverage section present but empty → Error

### Breaking Changes
- **Major**: mala.yaml now required (no fallback to Python defaults)
- [TBD: Any API changes to ValidationSpec?]
- [TBD: Changes to CLI arguments?]

## Testing & Validation

### Test Categories
- Unit tests for each new module (Tasks 1-6)
- Unit test updates for modified modules (Tasks 7-12)
- Integration tests (Task 13)
- [TBD: E2E tests with real language projects?]

### Acceptance Criteria Coverage
| Spec AC | Covered By |
|---------|------------|
| Go project with valid mala.yaml executes Go commands | Task 7, Task 13 |
| Node.js project executes npm commands | Task 7, Task 13 |
| Preset with custom test override | Task 4, Task 7, Task 13 |
| Only setup and test defined → skips undefined steps | Task 7, Task 13 |
| No mala.yaml → fails fast with clear error | Task 2, Task 12 |
| Invalid YAML → fails fast with specific error | Task 2 |
| Coverage with format: xml enforces threshold | Task 8, Task 13 |
| No coverage section → no coverage evaluation | Task 8, Task 13 |
| Custom code_patterns triggers validation | Task 6, Task 7, Task 13 |
| Tool name extraction skips wrappers | Task 5 |

## Rollback Strategy (Plan-Level)
- [TBD: Feature flag to toggle old vs new behavior?]
- [TBD: Phased rollout approach?]
- Each task has individual rollback steps (delete new files, revert changes)
- Full rollback: Revert all changes to modified files, delete new files/directories

## Open Questions
1. Should there be a transition period with fallback to Python defaults?
2. Is eager vs lazy config loading preferred?
3. How should the preset directory be structured for package distribution?
4. Should we add a `--config` CLI flag to specify alternate config path?
5. What happens if coverage.command fails but test passes?
6. Should we support command timeouts in config?
7. How to handle Windows shell differences (mentioned in spec as limitation)?
