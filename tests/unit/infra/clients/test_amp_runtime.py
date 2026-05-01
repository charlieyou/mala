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

    Per :data:`src.core.protocols.sdk.McpServerFactory`, the factory returns a
    server *map* keyed by server name — not the full Amp ``--mcp-config``
    payload. ``AmpRuntimeBuilder.build()`` is responsible for wrapping the map
    under the required ``{"mcpServers": ...}`` envelope.
    """

    def factory(
        agent_id: str,
        repo_path: Path,
        emit_lock_event: object,
    ) -> dict[str, object]:
        del emit_lock_event
        return {
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
        return {"locking_mcp": {"command": "x", "args": []}}

    AmpRuntimeBuilder(repo_path, "agent-77", factory).build()
    assert seen["agent_id"] == "agent-77"
    assert seen["repo_path"] == repo_path
    assert seen["emit_lock_event"] is None


@pytest.mark.unit
def test_mcp_config_wraps_factory_map_in_mcpservers_envelope(
    repo_path: Path,
) -> None:
    """The Amp ``--mcp-config`` JSON requires a top-level ``mcpServers`` key.

    The :data:`McpServerFactory` protocol returns a server *map* (e.g.
    ``{"mala-locking": <spec>}``), not a complete Amp config. If the builder
    forwarded the map straight to ``--mcp-config``, Amp would parse JSON with
    no ``mcpServers`` key and start with no MCP servers — silently dropping
    ``locking_mcp`` and breaking lock enforcement (fail-open). The builder must
    wrap the factory result.
    """

    def server_map_factory(
        agent_id: str, rp: Path, emit_lock_event: object
    ) -> dict[str, object]:
        del agent_id, rp, emit_lock_event
        return {"locking_mcp": {"command": "x", "args": []}}

    runtime = AmpRuntimeBuilder(repo_path, "a", server_map_factory).build()

    # mcp_config has the envelope.
    assert "mcpServers" in runtime.mcp_config
    assert runtime.mcp_config["mcpServers"] == {
        "locking_mcp": {"command": "x", "args": []}
    }
    # The serialized payload Amp actually sees on argv has the envelope too.
    idx = runtime.argv.index("--mcp-config")
    parsed = json.loads(runtime.argv[idx + 1])
    assert "mcpServers" in parsed
    assert "locking_mcp" in parsed["mcpServers"]


@pytest.mark.unit
def test_build_rejects_non_serializable_factory_with_actionable_error(
    repo_path: Path,
) -> None:
    """A Claude-path ``McpServerFactory`` returns server-config dicts containing
    in-process Claude SDK ``instance`` objects that are not JSON-serializable.
    If misrouted into ``AmpRuntimeBuilder``, ``json.dumps`` would raise an
    opaque ``TypeError`` deep inside ``subprocess.Popen``. The builder must
    surface a contract-explicit error at the seam instead."""

    class _UnSerializable:
        """Stand-in for the Claude SDK's in-process server instance."""

    def claude_shaped_factory(
        agent_id: str, rp: Path, emit_lock_event: object
    ) -> dict[str, object]:
        del agent_id, rp, emit_lock_event
        # Mirrors the Claude path's create_locking_mcp_server() return shape:
        # {"mala-locking": {"type": "sdk", "name": ..., "instance": <SDK obj>}}
        return {
            "mala-locking": {
                "type": "sdk",
                "name": "mala-locking",
                "instance": _UnSerializable(),
            }
        }

    builder = AmpRuntimeBuilder(repo_path, "agent-1", claude_shaped_factory)
    with pytest.raises(TypeError) as exc_info:
        builder.build()
    msg = str(exc_info.value)
    assert "Amp" in msg
    assert "stdio" in msg.lower()
    assert "Claude" in msg


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


@pytest.mark.unit
def test_pending_log_paths_are_unique_per_fresh_session(repo_path: Path) -> None:
    """Concurrent fresh Amp sessions must not share the same pending tee file.

    Mala spawns multiple issue agents concurrently into the shared
    ``~/.config/mala/amp-sessions/`` directory. If two ``build()`` calls
    returned the same pending log path, both AmpClient subprocesses would
    tee stream-json bytes into the same file, interleaving (and after
    ``system(init)``, racing on the rename to ``{thread_id}.jsonl``)."""
    paths = {
        AmpRuntimeBuilder(repo_path, f"agent-{i}", _stdio_locking_factory())
        .build()
        .log_path
        for i in range(64)
    }
    assert len(paths) == 64


@pytest.mark.unit
def test_pending_log_path_shape_is_pending_uuid_jsonl(
    builder: AmpRuntimeBuilder,
) -> None:
    runtime = builder.build()
    name = runtime.log_path.name
    assert name.startswith(".pending-")
    assert name.endswith(".jsonl")
    # uuid4().hex is 32 hex chars between the prefix and suffix
    middle = name[len(".pending-") : -len(".jsonl")]
    assert len(middle) == 32
    assert all(c in "0123456789abcdef" for c in middle)


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


# ---------------------------------------------------------------------------
# Fluent-API regression: pipeline calls
# builder.with_hooks().with_env().with_mcp().with_disallowed_tools()
# .with_lint_tools().build() and reads runtime.options + runtime.lint_cache.
# Without these, AgentSessionRunner._build_session raises AttributeError
# on the first Amp issue session (AC#6 break documented at
# plans/2026-04-29-amp-provider-plan.md#L165).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fluent_chain_used_by_agent_session_runner_returns_runtime(
    builder: AmpRuntimeBuilder,
) -> None:
    """The exact chain ``AgentSessionRunner._build_session`` calls must
    produce an :class:`AmpRuntime`.

    Mirrors ``src/pipeline/agent_session_runner.py:426-433``. A regression
    that drops one of these methods would re-introduce the AttributeError
    the reviewer flagged.
    """
    runtime = (
        builder.with_hooks(deadlock_monitor=None)
        .with_env(extra={"MALA_SDK_FLOW": "implementer"})
        .with_mcp()
        .with_disallowed_tools()
        .with_lint_tools(frozenset({"ruff"}))
        .build()
    )
    assert isinstance(runtime, AmpRuntime)


@pytest.mark.unit
def test_fluent_chain_used_by_fixer_service_returns_runtime(
    builder: AmpRuntimeBuilder,
) -> None:
    """Mirrors :meth:`FixerService._spawn_fixer`'s shaping (with explicit
    ``include_*`` flags + a manual ``servers={}`` override)."""
    runtime = (
        builder.with_hooks(
            deadlock_monitor=None,
            include_stop_hook=True,
            include_mala_disallowed_tools_hook=False,
        )
        .with_env(extra={"MALA_SDK_FLOW": "fixer"})
        .with_disallowed_tools()
        .with_mcp(servers={})
        .with_hooks(include_lock_enforcement_hook=False)
        .with_lint_tools(frozenset({"ruff"}))
        .build()
    )
    assert isinstance(runtime, AmpRuntime)
    # ``with_mcp(servers={})`` must clear the factory output so the fixer
    # path does not spawn locking-MCP under the empty-tools shape.
    assert runtime.mcp_config["mcpServers"] == {}


@pytest.mark.unit
def test_with_env_extras_are_layered_on_top_of_mandatory_overlays(
    builder: AmpRuntimeBuilder,
) -> None:
    """Caller extras (``MALA_SDK_FLOW`` etc.) reach the spawn env."""
    runtime = builder.with_env(extra={"MALA_SDK_FLOW": "implementer"}).build()
    assert runtime.env["MALA_SDK_FLOW"] == "implementer"
    # Mandatory overlays still in place — extras must not have clobbered them.
    assert runtime.env["PLUGINS"] == "all"
    assert runtime.env["MALA_AGENT_ID"] == "agent-42"


@pytest.mark.unit
def test_runtime_exposes_options_consumed_by_client_factory(
    builder: AmpRuntimeBuilder,
) -> None:
    """``runtime.options`` is what the pipeline forwards to
    ``client_factory.create``. Must be an :class:`AmpClientOptions` whose
    cwd / env / argv / log_path mirror the runtime's top-level fields."""
    from src.infra.clients.amp_client import AmpClientOptions

    runtime = builder.build()
    assert isinstance(runtime.options, AmpClientOptions)
    assert runtime.options.cwd == runtime.cwd
    assert dict(runtime.options.env) == dict(runtime.env)
    assert tuple(runtime.options.argv) == runtime.argv
    assert runtime.options.log_path == runtime.log_path
    assert runtime.options.thread_id == runtime.resume_thread_id


@pytest.mark.unit
def test_runtime_exposes_lint_cache_for_session_config(
    builder: AmpRuntimeBuilder,
) -> None:
    """``SessionConfig`` reads ``runtime.lint_cache`` unconditionally."""
    from src.infra.hooks import LintCache

    runtime = builder.with_lint_tools(frozenset({"ruff", "ty"})).build()
    assert isinstance(runtime.lint_cache, LintCache)
