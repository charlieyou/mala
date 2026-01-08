"""Unit tests for SDK adapter settings."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, cast

import pytest

from src.infra.sdk_adapter import SDKClientFactory

if TYPE_CHECKING:
    from claude_agent_sdk.types import ClaudeAgentOptions


@pytest.fixture
def factory() -> SDKClientFactory:
    """Create an SDKClientFactory instance for testing."""
    return SDKClientFactory()


class TestSettingSources:
    """Tests for setting_sources behavior in create_options."""

    def test_default_setting_sources(
        self, factory: SDKClientFactory, tmp_path: Path
    ) -> None:
        """Default setting_sources is ['local', 'project'] when not specified."""
        options = cast("ClaudeAgentOptions", factory.create_options(cwd=str(tmp_path)))
        assert options.setting_sources == ["local", "project"]

    def test_default_setting_sources_explicit_none(
        self, factory: SDKClientFactory, tmp_path: Path
    ) -> None:
        """Explicit None for setting_sources uses default ['local', 'project']."""
        options = cast(
            "ClaudeAgentOptions",
            factory.create_options(cwd=str(tmp_path), setting_sources=None),
        )
        assert options.setting_sources == ["local", "project"]

    def test_override_setting_sources(
        self, factory: SDKClientFactory, tmp_path: Path
    ) -> None:
        """Explicit setting_sources overrides the default."""
        options = cast(
            "ClaudeAgentOptions",
            factory.create_options(cwd=str(tmp_path), setting_sources=["user"]),
        )
        assert options.setting_sources == ["user"]

    def test_override_setting_sources_multiple(
        self, factory: SDKClientFactory, tmp_path: Path
    ) -> None:
        """Multiple sources can be provided as override."""
        options = cast(
            "ClaudeAgentOptions",
            factory.create_options(
                cwd=str(tmp_path), setting_sources=["project", "user"]
            ),
        )
        assert options.setting_sources == ["project", "user"]

    def test_override_setting_sources_empty_list(
        self, factory: SDKClientFactory, tmp_path: Path
    ) -> None:
        """Empty list preserves explicit override (no defaults)."""
        options = cast(
            "ClaudeAgentOptions",
            factory.create_options(cwd=str(tmp_path), setting_sources=[]),
        )
        assert options.setting_sources == []
