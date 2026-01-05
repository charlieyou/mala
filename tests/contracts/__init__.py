"""Contract tests for fake-vs-real implementation parity.

Contract tests validate that fake implementations behave identically to their
real counterparts and that fakes fully implement required protocols.

Run contract tests:
    uv run pytest tests/contracts/ -v
"""


def get_protocol_members(protocol_cls: type) -> frozenset[str]:
    """Get the declared members of a Protocol class.

    Uses typing.get_protocol_members when available (Python 3.13+),
    otherwise falls back to typing_extensions.get_protocol_members.

    Args:
        protocol_cls: A Protocol class to inspect.

    Returns:
        Frozenset of member names declared by the protocol.
    """
    try:
        from typing import get_protocol_members as _get_members  # type: ignore[attr-defined]
    except ImportError:
        from typing_extensions import get_protocol_members as _get_members

    return _get_members(protocol_cls)
