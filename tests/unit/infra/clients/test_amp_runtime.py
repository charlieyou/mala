"""Unit tests for :class:`AmpRuntime` + :class:`AmpRuntimeBuilder`.

Covers the runtime-builder slice of plan section "Testing & Validation"
(``plans/2026-04-29-amp-provider-plan.md#L751-L762``):

- CLI args include ``--execute --stream-json --dangerously-allow-all
  --mcp-config <json>``.
- Env includes ``PLUGINS=all`` and ``AMP_API_KEY`` passthrough.
- **Regression guard**: env includes ``MALA_AGENT_ID``, ``MALA_LOCK_DIR``,
  ``MALA_REPO_NAMESPACE`` with values matching what the Claude path's
  ``AgentRuntimeBuilder.with_env()`` would inject for ``AGENT_ID``,
  ``LOCK_DIR``, ``REPO_NAMESPACE`` — if any of the three is missing, the
  ``mala-safety.ts`` plugin's lock-ownership check is silently disabled,
  so this is asserted positively *and* defensively.
- ``--mode <smart|rush|deep>`` wired from :class:`AmpOptions.mode`.
- ``mcp_config`` JSON contains the ``locking_mcp`` stdio launch spec.
- Env is built from ``os.environ`` (not a bare dict).
- Config-override chain: ``MalaConfig`` loaded with
  ``coder_options.amp.mode='deep'`` flows through to ``AmpRuntime.argv``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, cast

import pytest

from src.core.protocols.agent_provider import CoderRuntimeBuilder
from src.infra.agent_runtime import AgentRuntimeBuilder
from src.infra.clients.amp_runtime import AmpRuntime, AmpRuntimeBuilder
from src.infra.io.config import MalaConfig
from src.infra.tools.env import SCRIPTS_DIR
from tests.fakes import FakeSDKClientFactory

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _stdio_locking_factory() -> Callable[..., dict[str, object]]:
    """Fake ``McpServerFactory`` returning a stdio-shaped ``locking_mcp`` spec.

    The Amp ``--mcp-config`` payload is a stdio launch description (command,
    args, env), unlike the Claude path's SDK-shaped server. The Amp orchestration
    code wires a stdio factory; here we mimic its output shape.
    """

    def factory(
        agent_id: str,
        repo_path: Path,
        emit_lock_event: object,
    ) -> dict[str, object]:
        del emit_lock_event
        return {
            "mcpServers": {
                "locking_mcp": {
                    "command": "uv",
                    "args": [
                        "run",
                        "python",
                        "-m",
                        "src.infra.tools.locking_mcp_stdio",
                        "--agent-id",
                        agent_id,
                        "--repo-namespace",
                        str(repo_path),
                    ],
                    "env": {},
                }
            }
        }

    return factory


@pytest.fixture
def repo_path(tmp_path: Path) -> Path:
    return tmp_path / "repo"


@pytest.fixture
def builder(repo_path: Path) -> AmpRuntimeBuilder:
    return AmpRuntimeBuilder(
        repo_path,
        "agent-42",
        _stdio_locking_factory(),
        mode="smart",
    )


# ---------------------------------------------------------------------------
# Conformance to the CoderRuntimeBuilder protocol.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_builder_conforms_to_coder_runtime_builder_protocol(
    builder: AmpRuntimeBuilder,
) -> None:
    assert isinstance(builder, CoderRuntimeBuilder)


@pytest.mark.unit
def test_build_returns_amp_runtime(builder: AmpRuntimeBuilder, repo_path: Path) -> None:
    runtime = builder.build()
    assert isinstance(runtime, AmpRuntime)
    assert runtime.cwd == repo_path


# ---------------------------------------------------------------------------
# argv assembly: --execute --stream-json --dangerously-allow-all --mcp-config
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_argv_starts_with_amp_executable(builder: AmpRuntimeBuilder) -> None:
    runtime = builder.build()
    assert runtime.argv[0] == "amp"


@pytest.mark.unit
def test_argv_contains_required_flags(builder: AmpRuntimeBuilder) -> None:
    runtime = builder.build()
    assert "--execute" in runtime.argv
    assert "--stream-json" in runtime.argv
    assert "--dangerously-allow-all" in runtime.argv
    assert "--mcp-config" in runtime.argv


@pytest.mark.unit
def test_argv_mcp_config_value_is_json_of_factory_dict(
    builder: AmpRuntimeBuilder,
) -> None:
    runtime = builder.build()
    idx = runtime.argv.index("--mcp-config")
    payload = runtime.argv[idx + 1]
    assert json.loads(payload) == runtime.mcp_config


# ---------------------------------------------------------------------------
# AC#4: --mode wired from AmpOptions.mode (smart, rush, deep).
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["smart", "rush", "deep"])
def test_mode_flows_into_argv_and_runtime_field(repo_path: Path, mode: str) -> None:
    runtime = AmpRuntimeBuilder(
        repo_path,
        "agent-1",
        _stdio_locking_factory(),
        mode=cast('Literal["smart", "rush", "deep"]', mode),
    ).build()
    assert runtime.mode == mode
    idx = runtime.argv.index("--mode")
    assert runtime.argv[idx + 1] == mode


@pytest.mark.unit
def test_default_mode_is_smart_when_omitted(
    repo_path: Path,
) -> None:
    runtime = AmpRuntimeBuilder(repo_path, "agent-1", _stdio_locking_factory()).build()
    assert runtime.mode == "smart"
    idx = runtime.argv.index("--mode")
    assert runtime.argv[idx + 1] == "smart"


# ---------------------------------------------------------------------------
# Env composition: built from os.environ + overlays (never a bare dict).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_env_inherits_os_environ_keys(
    builder: AmpRuntimeBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", "/tmp/fake-home")
    monkeypatch.setenv("TMPDIR", "/tmp/fake-tmp")
    monkeypatch.setenv("LANG", "C.UTF-8")
    runtime = builder.build()
    assert runtime.env["HOME"] == "/tmp/fake-home"
    assert runtime.env["TMPDIR"] == "/tmp/fake-tmp"
    assert runtime.env["LANG"] == "C.UTF-8"


@pytest.mark.unit
def test_env_path_prepends_scripts_dir(
    builder: AmpRuntimeBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    runtime = builder.build()
    assert runtime.env["PATH"] == f"{SCRIPTS_DIR}:/usr/bin:/bin"


@pytest.mark.unit
def test_env_includes_plugins_all(builder: AmpRuntimeBuilder) -> None:
    runtime = builder.build()
    assert runtime.env["PLUGINS"] == "all"


@pytest.mark.unit
def test_env_passes_through_amp_api_key(
    builder: AmpRuntimeBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AMP_API_KEY", "sk-amp-test-12345")
    runtime = builder.build()
    assert runtime.env["AMP_API_KEY"] == "sk-amp-test-12345"


@pytest.mark.unit
def test_env_includes_mcp_timeout_300000(builder: AmpRuntimeBuilder) -> None:
    runtime = builder.build()
    assert runtime.env["MCP_TIMEOUT"] == "300000"


# ---------------------------------------------------------------------------
# Regression guard: lock-ownership env vars must be present and match what
# the Claude path injects under different (unprefixed) names. Missing any
# silently disables the mala-safety.ts plugin's lock-ownership check.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "var", ["MALA_AGENT_ID", "MALA_LOCK_DIR", "MALA_REPO_NAMESPACE"]
)
def test_env_includes_each_lock_ownership_var(
    builder: AmpRuntimeBuilder, var: str
) -> None:
    runtime = builder.build()
    assert var in runtime.env, (
        f"{var} missing from AmpRuntime.env — would silently disable "
        "the mala-safety.ts lock-ownership check"
    )
    assert runtime.env[var] != ""


@pytest.mark.unit
def test_lock_ownership_env_values_match_claude_path(
    repo_path: Path,
) -> None:
    """The Amp ``MALA_*`` overlays must carry the same values the Claude path
    puts under unprefixed names. A divergence here means an Amp run computes
    different lock keys than the Python writer, breaking cross-language
    coordination."""
    repo_path.mkdir(parents=True, exist_ok=True)
    claude_runtime = (
        AgentRuntimeBuilder(repo_path, "agent-42", FakeSDKClientFactory())
        .with_env()
        .with_mcp(servers={})
        .build()
    )
    amp_runtime = AmpRuntimeBuilder(
        repo_path,
        "agent-42",
        _stdio_locking_factory(),
        mode="smart",
    ).build()

    assert amp_runtime.env["MALA_AGENT_ID"] == claude_runtime.env["AGENT_ID"]
    assert amp_runtime.env["MALA_LOCK_DIR"] == claude_runtime.env["LOCK_DIR"]
    assert (
        amp_runtime.env["MALA_REPO_NAMESPACE"] == claude_runtime.env["REPO_NAMESPACE"]
    )


# ---------------------------------------------------------------------------
# mcp_config: contains the locking_mcp stdio launch spec.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mcp_config_contains_locking_mcp_stdio_spec(
    builder: AmpRuntimeBuilder,
) -> None:
    runtime = builder.build()
    raw_servers = runtime.mcp_config["mcpServers"]
    assert isinstance(raw_servers, dict)
    servers = cast("dict[str, object]", raw_servers)
    assert "locking_mcp" in servers
    raw_spec = servers["locking_mcp"]
    assert isinstance(raw_spec, dict)
    spec = cast("dict[str, object]", raw_spec)
    # Stdio shape: command + args (env optional).
    assert "command" in spec
    assert "args" in spec


@pytest.mark.unit
def test_mcp_factory_receives_agent_id_and_repo_path(repo_path: Path) -> None:
    seen: dict[str, object] = {}

    def factory(agent_id: str, rp: Path, emit_lock_event: object) -> dict[str, object]:
        seen["agent_id"] = agent_id
        seen["repo_path"] = rp
        seen["emit_lock_event"] = emit_lock_event
        return {"mcpServers": {"locking_mcp": {"command": "x", "args": []}}}

    AmpRuntimeBuilder(repo_path, "agent-77", factory).build()
    assert seen["agent_id"] == "agent-77"
    assert seen["repo_path"] == repo_path
    assert seen["emit_lock_event"] is None


# ---------------------------------------------------------------------------
# Resume: with_resume(thread_id) flows into argv + runtime field + log_path.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_with_resume_sets_resume_thread_id(builder: AmpRuntimeBuilder) -> None:
    runtime = builder.with_resume("T-abc123").build()
    assert runtime.resume_thread_id == "T-abc123"


@pytest.mark.unit
def test_with_resume_returns_builder_for_chaining(
    builder: AmpRuntimeBuilder,
) -> None:
    assert builder.with_resume("T-xyz") is builder


@pytest.mark.unit
def test_with_resume_threads_thread_id_into_argv(
    builder: AmpRuntimeBuilder,
) -> None:
    runtime = builder.with_resume("T-deadbeef").build()
    assert "--thread-id" in runtime.argv
    idx = runtime.argv.index("--thread-id")
    assert runtime.argv[idx + 1] == "T-deadbeef"


@pytest.mark.unit
def test_no_resume_means_no_thread_id_in_argv(builder: AmpRuntimeBuilder) -> None:
    runtime = builder.build()
    assert "--thread-id" not in runtime.argv
    assert runtime.resume_thread_id is None


@pytest.mark.unit
def test_resume_log_path_uses_thread_id_filename(
    builder: AmpRuntimeBuilder,
) -> None:
    runtime = builder.with_resume("T-abc123").build()
    assert runtime.log_path.name == "T-abc123.jsonl"


# ---------------------------------------------------------------------------
# Config override chain: MalaConfig → AmpOptions.mode → AmpRuntime.argv.
# Validates T002 → T012 wiring without reaching for the dataclass directly.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_malaconfig_amp_mode_deep_flows_into_argv(
    repo_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load MalaConfig via the normal ``from_env`` path with a yaml-supplied
    deep mode (no env, no CLI), then build an AmpRuntime from the resolved
    ``coder_options.amp.mode``. The resulting argv must carry ``deep``."""
    for var in ("MALA_AMP_MODE", "MALA_CODER"):
        monkeypatch.delenv(var, raising=False)
    config = MalaConfig.from_env(validate=False, yaml_amp_mode="deep")
    assert config.coder_options.amp.mode == "deep"

    runtime = AmpRuntimeBuilder(
        repo_path,
        "agent-1",
        _stdio_locking_factory(),
        mode=config.coder_options.amp.mode,
    ).build()

    idx = runtime.argv.index("--mode")
    assert runtime.argv[idx + 1] == "deep"
    assert runtime.mode == "deep"
