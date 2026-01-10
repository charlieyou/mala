"""Unit tests for Anthropic client factory."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    import types
    from collections.abc import Mapping, Sequence


@pytest.mark.unit
class TestCreateAnthropicClient:
    """Tests for the create_anthropic_client factory function."""

    def test_creates_client_with_defaults(self) -> None:
        """Creates client with no explicit args (uses env vars)."""
        mock_anthropic_class = MagicMock()
        with patch.dict(
            "sys.modules", {"anthropic": MagicMock(Anthropic=mock_anthropic_class)}
        ):
            # Re-import to pick up the mock
            from importlib import reload

            import src.infra.clients.anthropic_client as module

            reload(module)
            module.create_anthropic_client()

        mock_anthropic_class.assert_called_once_with()

    def test_creates_client_with_api_key(self) -> None:
        """Creates client with explicit api_key."""
        mock_anthropic_class = MagicMock()
        with patch.dict(
            "sys.modules", {"anthropic": MagicMock(Anthropic=mock_anthropic_class)}
        ):
            from importlib import reload

            import src.infra.clients.anthropic_client as module

            reload(module)
            module.create_anthropic_client(api_key="sk-test-key")

        mock_anthropic_class.assert_called_once_with(api_key="sk-test-key")

    def test_creates_client_with_base_url(self) -> None:
        """Creates client with explicit base_url."""
        mock_anthropic_class = MagicMock()
        with patch.dict(
            "sys.modules", {"anthropic": MagicMock(Anthropic=mock_anthropic_class)}
        ):
            from importlib import reload

            import src.infra.clients.anthropic_client as module

            reload(module)
            module.create_anthropic_client(base_url="https://proxy.example.com/v1")

        mock_anthropic_class.assert_called_once_with(
            base_url="https://proxy.example.com/v1"
        )

    def test_creates_client_with_timeout(self) -> None:
        """Creates client with explicit timeout."""
        mock_anthropic_class = MagicMock()
        with patch.dict(
            "sys.modules", {"anthropic": MagicMock(Anthropic=mock_anthropic_class)}
        ):
            from importlib import reload

            import src.infra.clients.anthropic_client as module

            reload(module)
            module.create_anthropic_client(timeout=60.0)

        mock_anthropic_class.assert_called_once_with(timeout=60.0)

    def test_creates_client_with_all_params(self) -> None:
        """Creates client with all params specified."""
        mock_anthropic_class = MagicMock()
        with patch.dict(
            "sys.modules", {"anthropic": MagicMock(Anthropic=mock_anthropic_class)}
        ):
            from importlib import reload

            import src.infra.clients.anthropic_client as module

            reload(module)
            module.create_anthropic_client(
                api_key="sk-test",
                base_url="https://proxy.example.com",
                timeout=120.0,
            )

        mock_anthropic_class.assert_called_once_with(
            api_key="sk-test",
            base_url="https://proxy.example.com",
            timeout=120.0,
        )

    def test_raises_runtime_error_when_anthropic_not_installed(self) -> None:
        """Raises RuntimeError when anthropic package is not installed."""
        import sys

        # Remove cached module
        if "src.infra.clients.anthropic_client" in sys.modules:
            del sys.modules["src.infra.clients.anthropic_client"]

        # Now import fresh
        from src.infra.clients import anthropic_client

        # Patch the import inside the function to fail
        original_import = builtins.__import__

        def mock_import(
            name: str,
            globals: Mapping[str, object] | None = None,
            locals: Mapping[str, object] | None = None,
            fromlist: Sequence[str] | None = (),
            level: int = 0,
        ) -> types.ModuleType:
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return original_import(name, globals, locals, fromlist, level)

        with patch.object(builtins, "__import__", mock_import):
            with pytest.raises(RuntimeError) as exc_info:
                anthropic_client.create_anthropic_client()

            assert "anthropic package is required" in str(exc_info.value)
