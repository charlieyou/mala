"""Contract tests for EnvConfigPort implementations.

Ensures FakeEnvConfig implements all methods of EnvConfigPort protocol.
"""

from pathlib import Path

import pytest

from src.core.protocols.infra import EnvConfigPort
from tests.contracts import get_protocol_members
from tests.fakes.env_config import FakeEnvConfig


@pytest.mark.unit
def test_fake_env_config_implements_all_protocol_methods() -> None:
    """FakeEnvConfig must implement all public methods of EnvConfigPort."""
    protocol_methods = get_protocol_members(EnvConfigPort)
    fake_methods = {name for name in dir(FakeEnvConfig) if not name.startswith("_")}

    missing = protocol_methods - fake_methods
    assert not missing, f"FakeEnvConfig missing protocol methods: {sorted(missing)}"


@pytest.mark.unit
def test_fake_env_config_protocol_compliance() -> None:
    """FakeEnvConfig passes runtime isinstance check for EnvConfigPort."""
    config = FakeEnvConfig()
    assert isinstance(config, EnvConfigPort)


class TestFakeEnvConfigBehavior:
    """Behavioral tests for FakeEnvConfig."""

    @pytest.mark.unit
    def test_scripts_dir_returns_path(self) -> None:
        """scripts_dir property returns a Path."""
        config = FakeEnvConfig()
        assert isinstance(config.scripts_dir, Path)

    @pytest.mark.unit
    def test_cache_dir_returns_path(self) -> None:
        """cache_dir property returns a Path."""
        config = FakeEnvConfig()
        assert isinstance(config.cache_dir, Path)

    @pytest.mark.unit
    def test_lock_dir_returns_path(self) -> None:
        """lock_dir property returns a Path."""
        config = FakeEnvConfig()
        assert isinstance(config.lock_dir, Path)

    @pytest.mark.unit
    def test_find_cerberus_bin_path_returns_none_by_default(self) -> None:
        """find_cerberus_bin_path() returns None by default."""
        config = FakeEnvConfig()
        assert config.find_cerberus_bin_path() is None

    @pytest.mark.unit
    def test_find_cerberus_bin_path_returns_configured_path(self) -> None:
        """find_cerberus_bin_path() returns configured path when set."""
        config = FakeEnvConfig(_cerberus_bin_path=Path("/usr/local/bin/cerberus"))
        result = config.find_cerberus_bin_path()
        assert result == Path("/usr/local/bin/cerberus")

    @pytest.mark.unit
    def test_paths_can_be_customized(self) -> None:
        """All paths can be customized via constructor."""
        config = FakeEnvConfig(
            _scripts_dir=Path("/custom/scripts"),
            _cache_dir=Path("/custom/cache"),
            _lock_dir=Path("/custom/locks"),
        )
        assert config.scripts_dir == Path("/custom/scripts")
        assert config.cache_dir == Path("/custom/cache")
        assert config.lock_dir == Path("/custom/locks")
