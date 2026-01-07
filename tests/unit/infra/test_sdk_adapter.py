"""Unit tests for SDK adapter settings."""

from __future__ import annotations

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

    def test_default_setting_sources(self, factory: SDKClientFactory) -> None:
        """Default setting_sources is ['local', 'project'] when not specified."""
        options = cast("ClaudeAgentOptions", factory.create_options(cwd="/tmp"))
        assert options.setting_sources == ["local", "project"]

    def test_default_setting_sources_explicit_none(
        self, factory: SDKClientFactory
    ) -> None:
        """Explicit None for setting_sources uses default ['local', 'project']."""
        options = cast(
            "ClaudeAgentOptions",
            factory.create_options(cwd="/tmp", setting_sources=None),
        )
        assert options.setting_sources == ["local", "project"]

    def test_override_setting_sources(self, factory: SDKClientFactory) -> None:
        """Explicit setting_sources overrides the default."""
        options = cast(
            "ClaudeAgentOptions",
            factory.create_options(cwd="/tmp", setting_sources=["user"]),
        )
        assert options.setting_sources == ["user"]

    def test_override_setting_sources_multiple(self, factory: SDKClientFactory) -> None:
        """Multiple sources can be provided as override."""
        options = cast(
            "ClaudeAgentOptions",
            factory.create_options(cwd="/tmp", setting_sources=["project", "user"]),
        )
        assert options.setting_sources == ["project", "user"]
