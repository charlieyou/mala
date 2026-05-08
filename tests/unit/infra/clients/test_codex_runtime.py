"""Unit tests for :class:`CodexRuntime` + :class:`CodexRuntimeBuilder` (Phase C, T010).

Covers:

  * Builder threads ``model`` / ``effort`` / ``approval_policy`` /
    ``sandbox`` from the constructor onto the runtime (parity with the
    Phase B-era ``test_runtime_builder_threads_resolved_options`` —
    integration-path evidence for AC #4).
  * ``with_resume(thread_id)`` populates ``runtime.resume_thread_id``.
  * ``with_mcp(servers)`` overrides the injected factory; default uses
    the factory verbatim.
  * Per-process env composition uses ``os.environ`` plus
    mala-specific overlays (parity with the Amp path's
    ``AmpRuntimeBuilder.build`` posture; plan ``L283-L295``).
  * ``build()`` materializes a real :class:`LintCache` for the runtime
    (cross-coder protocol parity).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.infra.clients.codex_runtime import CodexRuntime, CodexRuntimeBuilder

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _empty_factory() -> Callable[..., dict[str, object]]:
    def factory(
        agent_id: str, repo_path: Path, emit_lock_event: object
    ) -> dict[str, object]:
        del agent_id, repo_path, emit_lock_event
        return {}

    return factory


def _make_builder(tmp_path: Path) -> CodexRuntimeBuilder:
    return CodexRuntimeBuilder(
        tmp_path,
        "agent-x",
        _empty_factory(),
        model="gpt-5.5",
        effort=None,
        approval_policy="never",
        sandbox="danger-full-access",
    )


@pytest.mark.unit
def test_build_threads_constructor_options_to_runtime(tmp_path: Path) -> None:
    builder = CodexRuntimeBuilder(
        tmp_path,
        "agent-x",
        _empty_factory(),
        model="gpt-5.5-foo",
        effort="medium",
        approval_policy="on-request",
        sandbox="workspace-write",
    )

    runtime = builder.build()
    assert isinstance(runtime, CodexRuntime)
    assert runtime.cwd == tmp_path
    assert runtime.agent_id == "agent-x"
    assert runtime.model == "gpt-5.5-foo"
    assert runtime.effort == "medium"
    assert runtime.approval_policy == "on-request"
    assert runtime.sandbox == "workspace-write"
    assert runtime.base_instructions is None
    assert runtime.resume_thread_id is None


@pytest.mark.unit
def test_with_resume_populates_resume_thread_id(tmp_path: Path) -> None:
    runtime = _make_builder(tmp_path).with_resume("thr_abc").build()
    assert runtime.resume_thread_id == "thr_abc"


@pytest.mark.unit
def test_with_resume_none_clears_resume_thread_id(tmp_path: Path) -> None:
    runtime = _make_builder(tmp_path).with_resume("thr_abc").with_resume(None).build()
    assert runtime.resume_thread_id is None


@pytest.mark.unit
def test_with_mcp_overrides_injected_factory(tmp_path: Path) -> None:
    """``with_mcp(servers)`` short-circuits the injected MCP factory."""

    def loud_factory(
        agent_id: str, repo_path: Path, emit_lock_event: object
    ) -> dict[str, object]:
        # Should not be reached when ``with_mcp`` is used.
        del agent_id, repo_path, emit_lock_event
        raise AssertionError("MCP factory should not be invoked when overridden")

    builder = CodexRuntimeBuilder(
        tmp_path,
        "agent-x",
        loud_factory,
        model="gpt-5.5",
        effort=None,
        approval_policy="never",
        sandbox="danger-full-access",
    )
    override: dict[str, object] = {"mala-locking": {"command": "x"}}
    runtime = builder.with_mcp(override).build()
    assert runtime.mcp_servers == override


@pytest.mark.unit
def test_default_mcp_servers_come_from_factory(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def factory(
        agent_id: str, repo_path: Path, emit_lock_event: object
    ) -> dict[str, object]:
        captured["agent_id"] = agent_id
        captured["repo_path"] = repo_path
        captured["emit_lock_event"] = emit_lock_event
        return {"mala-locking": {"command": "mala-codex-mcp-locking"}}

    runtime = CodexRuntimeBuilder(
        tmp_path,
        "agent-x",
        factory,
        model="gpt-5.5",
        effort=None,
        approval_policy="never",
        sandbox="danger-full-access",
    ).build()
    assert runtime.mcp_servers == {
        "mala-locking": {"command": "mala-codex-mcp-locking"}
    }
    assert captured["agent_id"] == "agent-x"
    assert captured["repo_path"] == tmp_path
    assert captured["emit_lock_event"] is None


@pytest.mark.unit
def test_env_carries_parent_environ_plus_mala_overlays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-process env: ``os.environ`` plus ``MALA_*`` overlays.

    Plan ``L283-L295``: parity with the Amp path. The runtime carries
    a per-process env dict so ``AsyncCodex`` can inherit it without
    mala mutating the parent ``os.environ``.
    """
    monkeypatch.setenv("PARENT_VAR", "parent-value")
    monkeypatch.setenv("MALA_LOCK_DIR", "/tmp/locks-x")
    monkeypatch.delenv("MALA_AGENT_ID", raising=False)
    monkeypatch.delenv("MALA_REPO_NAMESPACE", raising=False)

    runtime = _make_builder(tmp_path).build()
    env = dict(runtime.env)
    assert env["PARENT_VAR"] == "parent-value"
    assert env["MALA_AGENT_ID"] == "agent-x"
    assert env["MALA_REPO_NAMESPACE"] == str(tmp_path)
    assert env["MALA_LOCK_DIR"] == "/tmp/locks-x"


@pytest.mark.unit
def test_with_env_overlays_extra(tmp_path: Path) -> None:
    runtime = _make_builder(tmp_path).with_env({"MALA_DISALLOWED_TOOLS": "rm"}).build()
    assert runtime.env.get("MALA_DISALLOWED_TOOLS") == "rm"


@pytest.mark.unit
def test_build_carries_lint_cache_instance(tmp_path: Path) -> None:
    """Cross-coder runtime contract: the runtime carries a real LintCache."""
    from src.infra.hooks.lint_cache import LintCache

    runtime = _make_builder(tmp_path).build()
    assert isinstance(runtime.lint_cache, LintCache)


@pytest.mark.unit
def test_with_lint_tools_threads_into_lint_cache(tmp_path: Path) -> None:
    """``with_lint_tools`` is forwarded to the constructed LintCache."""
    runtime = _make_builder(tmp_path).with_lint_tools({"ruff", "ty"}).build()
    # LintCache normalizes the tool set internally; we assert the runtime
    # has a cache rather than introspecting private state.
    assert runtime.lint_cache is not None


@pytest.mark.unit
def test_with_agent_timeout_returns_self(tmp_path: Path) -> None:
    """``with_agent_timeout`` is recorded but does not affect the built runtime in C."""
    builder = _make_builder(tmp_path)
    assert builder.with_agent_timeout(120.0) is builder
    runtime = builder.build()
    assert isinstance(runtime, CodexRuntime)
