"""Unit tests for the shared plugin-provider base module (T_B4).

Drives :func:`install_with_selftest` and the public dataclass /
exception surface with fakes; the contract test in
``tests/contracts/test_plugin_provider_spec_contract.py`` exercises
the Protocol against the real Amp/Codex providers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.infra.clients._plugin_provider_base import (
    PluginInstallError,
    SelftestCache,
    StdioMcpLaunchSpec,
    install_with_selftest,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# install_with_selftest — happy path + cache short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runs_install_then_selftest_on_first_call() -> None:
    """First call must run the installer and then pass the hash to selftest."""
    call_order: list[str] = []
    selftest_args: list[str] = []

    def install() -> str:
        call_order.append("install")
        return "hash-abc123"

    def selftest(plugin_hash: str) -> None:
        call_order.append("selftest")
        selftest_args.append(plugin_hash)

    cache = SelftestCache()
    ran = install_with_selftest(install=install, selftest=selftest, cache=cache)

    assert ran is True
    assert call_order == ["install", "selftest"]
    assert selftest_args == ["hash-abc123"]
    assert cache.key == ("unknown", "hash-abc123")


@pytest.mark.unit
def test_short_circuits_when_cache_key_matches() -> None:
    """Second call with the same hash + version must skip the selftest."""
    selftest_calls: list[str] = []

    def install() -> str:
        return "hash-abc123"

    def selftest(plugin_hash: str) -> None:
        selftest_calls.append(plugin_hash)

    cache = SelftestCache()
    install_with_selftest(install=install, selftest=selftest, cache=cache)
    second_ran = install_with_selftest(install=install, selftest=selftest, cache=cache)

    assert second_ran is False
    assert selftest_calls == ["hash-abc123"]  # only the first call ran the probe


@pytest.mark.unit
def test_runs_selftest_again_when_plugin_hash_changes() -> None:
    """A new plugin hash must invalidate the cache and re-run the selftest."""
    hashes = iter(["hash-1", "hash-2"])
    selftest_calls: list[str] = []

    def install() -> str:
        return next(hashes)

    def selftest(plugin_hash: str) -> None:
        selftest_calls.append(plugin_hash)

    cache = SelftestCache()
    install_with_selftest(install=install, selftest=selftest, cache=cache)
    install_with_selftest(install=install, selftest=selftest, cache=cache)

    assert selftest_calls == ["hash-1", "hash-2"]
    assert cache.key == ("unknown", "hash-2")


@pytest.mark.unit
def test_runs_selftest_again_when_coder_version_changes() -> None:
    """A different coder_version must invalidate the cache."""
    selftest_calls: list[tuple[str, str]] = []

    def install() -> str:
        return "hash-abc123"

    def make_selftest(version: str) -> Callable[[str], None]:
        def selftest(plugin_hash: str) -> None:
            selftest_calls.append((version, plugin_hash))

        return selftest

    cache = SelftestCache()
    install_with_selftest(
        install=install,
        selftest=make_selftest("v1"),
        cache=cache,
        coder_version="v1",
    )
    install_with_selftest(
        install=install,
        selftest=make_selftest("v2"),
        cache=cache,
        coder_version="v2",
    )

    assert selftest_calls == [("v1", "hash-abc123"), ("v2", "hash-abc123")]
    assert cache.key == ("v2", "hash-abc123")


# ---------------------------------------------------------------------------
# install_with_selftest — fail-closed behavior
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_selftest_failure_propagates() -> None:
    """A raising selftest must propagate the error to the caller."""

    def install() -> str:
        return "hash-abc123"

    def selftest(plugin_hash: str) -> None:
        raise PluginInstallError("fake selftest failure")

    cache = SelftestCache()
    with pytest.raises(PluginInstallError, match="fake selftest failure"):
        install_with_selftest(install=install, selftest=selftest, cache=cache)


@pytest.mark.unit
def test_selftest_failure_does_not_cache() -> None:
    """A failing selftest must leave the cache unchanged so retries re-run it."""
    selftest_calls: list[str] = []
    raise_first_call = [True]

    def install() -> str:
        return "hash-abc123"

    def selftest(plugin_hash: str) -> None:
        selftest_calls.append(plugin_hash)
        if raise_first_call[0]:
            raise_first_call[0] = False
            raise PluginInstallError("flaky probe")

    cache = SelftestCache()
    with pytest.raises(PluginInstallError):
        install_with_selftest(install=install, selftest=selftest, cache=cache)
    assert cache.key is None

    # Retry: the cache is empty so selftest runs again and now succeeds.
    install_with_selftest(install=install, selftest=selftest, cache=cache)
    assert selftest_calls == ["hash-abc123", "hash-abc123"]
    assert cache.key == ("unknown", "hash-abc123")


@pytest.mark.unit
def test_install_failure_propagates() -> None:
    """A raising installer must propagate before the selftest fires."""
    selftest_calls: list[str] = []

    def install() -> str:
        raise PluginInstallError("install failed")

    def selftest(plugin_hash: str) -> None:
        selftest_calls.append(plugin_hash)

    cache = SelftestCache()
    with pytest.raises(PluginInstallError, match="install failed"):
        install_with_selftest(install=install, selftest=selftest, cache=cache)
    assert selftest_calls == []
    assert cache.key is None


# ---------------------------------------------------------------------------
# StdioMcpLaunchSpec
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stdio_mcp_launch_spec_renders_to_amp_codex_dict_shape() -> None:
    """``to_dict`` must produce the dict shape Amp/Codex MCP configs consume."""
    spec = StdioMcpLaunchSpec(
        command="mala-amp-mcp-locking",
        args=("--agent-id", "issue-1"),
        env={"MALA_AGENT_ID": "issue-1"},
    )
    payload = spec.to_dict()

    assert payload == {
        "command": "mala-amp-mcp-locking",
        "args": ["--agent-id", "issue-1"],
        "env": {"MALA_AGENT_ID": "issue-1"},
    }


@pytest.mark.unit
def test_stdio_mcp_launch_spec_defaults_are_empty() -> None:
    """A command-only spec must default to empty args/env (no None leakage)."""
    spec = StdioMcpLaunchSpec(command="mala-amp-mcp-locking")
    assert spec.args == ()
    assert dict(spec.env) == {}


# ---------------------------------------------------------------------------
# PluginInstallError
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plugin_install_error_is_runtime_error_subclass() -> None:
    """Plugin-install errors must be catchable as ``RuntimeError`` for parity
    with the existing ``AmpPluginNotActiveError`` / ``CodexHookNotActiveError``
    bases.
    """
    assert issubclass(PluginInstallError, RuntimeError)
