"""Protocol-shape tests for the cross-coder ``SDKClientFactoryProtocol``.

Asserts the cross-coder factory protocol exposes only the slim surface
shared by every coder backend (Claude, Amp, Codex): ``create(runtime)``
and ``with_resume(runtime, resume)``. Claude-only knobs
(``create_options``, ``create_hook_matcher``) must not appear on the
cross-coder protocol — they live on the Claude-private
``ClaudeSDKClientFactoryProtocol`` / ``SDKClientFactory``.

Pinning the shape here prevents Claude vocabulary from drifting back
onto the cross-coder protocol when new wiring is added (see plan AC#16).
"""

from __future__ import annotations

import inspect
import sys

import pytest

from src.core.protocols.sdk import SDKClientFactoryProtocol


CLAUDE_ONLY_METHODS = ("create_options", "create_hook_matcher")
SLIM_METHODS = ("create", "with_resume")


@pytest.mark.unit
@pytest.mark.parametrize("method_name", CLAUDE_ONLY_METHODS)
def test_no_claude_only_methods_on_cross_coder_protocol(method_name: str) -> None:
    """The cross-coder factory protocol must not carry Claude-only knobs."""
    assert not hasattr(SDKClientFactoryProtocol, method_name), (
        f"SDKClientFactoryProtocol.{method_name} leaks Claude vocabulary "
        "into the cross-coder protocol; move it onto the Claude-private "
        "factory class instead."
    )


@pytest.mark.unit
@pytest.mark.parametrize("method_name", SLIM_METHODS)
def test_slim_methods_present(method_name: str) -> None:
    """The slim cross-coder surface (``create`` + ``with_resume``) is required."""
    assert hasattr(SDKClientFactoryProtocol, method_name)


@pytest.mark.unit
def test_create_signature_takes_opaque_runtime() -> None:
    """``create`` accepts an opaque ``runtime`` argument; no Claude shape leaked."""
    sig = inspect.signature(SDKClientFactoryProtocol.create)
    params = [name for name in sig.parameters if name != "self"]
    # One positional parameter; name is part of the protocol contract
    # (``runtime``) so future implementations don't reintroduce
    # ``options=`` Claude vocabulary.
    assert params == ["runtime"]


@pytest.mark.unit
def test_with_resume_signature_takes_opaque_runtime() -> None:
    """``with_resume`` accepts ``(runtime, resume)`` — no Claude options shape."""
    sig = inspect.signature(SDKClientFactoryProtocol.with_resume)
    params = [name for name in sig.parameters if name != "self"]
    assert params == ["runtime", "resume"]


@pytest.mark.unit
def test_fake_factory_create_call_docs_use_runtime_name() -> None:
    """Regression: fake factory docs must match the runtime-shaped protocol."""
    from tests.fakes.sdk_client import FakeSDKClientFactory

    module = sys.modules[FakeSDKClientFactory.__module__]
    docs = "\n".join(
        doc
        for doc in (
            inspect.getdoc(module) or "",
            FakeSDKClientFactory.__doc__ or "",
        )
    )
    assert "runtimes passed to create" in docs
    assert "options passed to create" not in docs


@pytest.mark.unit
def test_amp_factory_satisfies_slim_protocol() -> None:
    """The Amp factory satisfies the slim protocol without needing walls.

    Regression for AC#16: the Amp factory previously had to raise
    ``NotImplementedError`` on Claude-only methods to satisfy the
    Claude-shaped protocol. With the protocol slimmed, it no longer needs
    those walls and conforms cleanly.
    """
    from src.infra.clients.amp_provider import _AmpClientFactory

    factory = _AmpClientFactory()
    assert isinstance(factory, SDKClientFactoryProtocol)
    # No Claude-only knobs hanging off the Amp factory either.
    for name in CLAUDE_ONLY_METHODS:
        assert not hasattr(factory, name), (
            f"_AmpClientFactory unexpectedly carries Claude knob {name!r}; "
            "the slim protocol no longer requires it."
        )


@pytest.mark.unit
def test_claude_factory_keeps_claude_knobs() -> None:
    """The Claude-private factory still exposes the Claude-only knobs.

    They moved off the cross-coder protocol but stay accessible on the
    concrete Claude factory class for Claude-side wiring (e.g. the
    ``AgentRuntimeBuilder``).
    """
    from src.infra.sdk_adapter import SDKClientFactory

    factory = SDKClientFactory()
    for name in (*SLIM_METHODS, *CLAUDE_ONLY_METHODS):
        assert callable(getattr(factory, name)), (
            f"Claude SDKClientFactory must still expose {name!r} for "
            "Claude-side wiring code."
        )
