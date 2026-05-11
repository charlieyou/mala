"""Contract tests for :class:`PluginProviderSpec` (T_B4).

Parametrized over the real ``AmpAgentProvider`` and ``CodexAgentProvider``
classes: both must satisfy the Protocol structurally, expose the
stdio-shaped MCP factory, and re-root their plugin-install error
classes onto the shared :class:`PluginInstallError` base.

The exception-base assertions are **expected to fail** until T_B5
(Amp provider migration) and T_B6 (Codex provider migration) re-root
``AmpPluginNotActiveError`` and ``CodexHookNotActiveError`` onto
:class:`PluginInstallError`. The contract test exists ahead of those
migrations so the migration PRs can flip from red to green.
"""

from __future__ import annotations

import pytest

from src.infra.clients._plugin_provider_base import (
    PluginInstallError,
    PluginProviderSpec,
)
from src.infra.clients.amp_provider import (
    AmpAgentProvider,
    AmpPluginNotActiveError,
)
from src.infra.clients.codex_provider import (
    CodexAgentProvider,
    CodexHookNotActiveError,
)


# ---------------------------------------------------------------------------
# Parametrization helpers
# ---------------------------------------------------------------------------


_PROVIDER_CASES = [
    pytest.param(AmpAgentProvider, AmpPluginNotActiveError, id="amp"),
    pytest.param(CodexAgentProvider, CodexHookNotActiveError, id="codex"),
]


# ---------------------------------------------------------------------------
# Structural Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(("provider_cls", "_error_cls"), _PROVIDER_CASES)
def test_provider_structurally_conforms_to_spec(
    provider_cls: type, _error_cls: type[Exception]
) -> None:
    """Both providers must satisfy the runtime-checkable PluginProviderSpec."""
    provider = provider_cls()
    assert isinstance(provider, PluginProviderSpec)


@pytest.mark.unit
@pytest.mark.parametrize(("provider_cls", "_error_cls"), _PROVIDER_CASES)
def test_provider_exposes_callable_mcp_server_factory(
    provider_cls: type, _error_cls: type[Exception]
) -> None:
    """``mcp_server_factory()`` must return a callable factory."""
    provider = provider_cls()
    factory = provider.mcp_server_factory()
    assert callable(factory)


@pytest.mark.unit
@pytest.mark.parametrize(("provider_cls", "_error_cls"), _PROVIDER_CASES)
def test_provider_name_is_amp_or_codex(
    provider_cls: type, _error_cls: type[Exception]
) -> None:
    """``name`` must be one of the plugin-gated coder identifiers."""
    provider = provider_cls()
    assert provider.name in ("amp", "codex")


# ---------------------------------------------------------------------------
# Exception unification (expected to fail until T_B5 / T_B6)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.xfail(
    reason=(
        "Re-rooting AmpPluginNotActiveError (T_B5) and CodexHookNotActiveError "
        "(T_B6) onto PluginInstallError is the documented migration target; "
        "the migration PRs will remove this marker."
    ),
    strict=False,
)
@pytest.mark.parametrize(("_provider_cls", "error_cls"), _PROVIDER_CASES)
def test_provider_install_error_inherits_from_plugin_install_error(
    _provider_cls: type, error_cls: type[Exception]
) -> None:
    """Provider-specific install errors must be PluginInstallError subclasses.

    Will fail until T_B5 re-roots ``AmpPluginNotActiveError`` and T_B6
    re-roots ``CodexHookNotActiveError`` onto :class:`PluginInstallError`
    (the documented migration target for Workstream B).
    """
    assert issubclass(error_cls, PluginInstallError), (
        f"{error_cls.__name__} must inherit from PluginInstallError "
        "(target after T_B5 / T_B6 migration)"
    )
