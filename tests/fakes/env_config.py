"""Fake environment config for testing."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FakeEnvConfig:
    """In-memory env config implementing EnvConfigPort.

    All paths default to /fake-* paths. Tests can override as needed.
    """

    _scripts_dir: Path = field(default_factory=lambda: Path("/fake-scripts"))
    _cache_dir: Path = field(default_factory=lambda: Path("/fake-cache"))
    _lock_dir: Path = field(default_factory=lambda: Path("/fake-locks"))
    _cerberus_bin_path: Path | None = None

    @property
    def scripts_dir(self) -> Path:
        return self._scripts_dir

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def lock_dir(self) -> Path:
        return self._lock_dir

    def find_cerberus_bin_path(self) -> Path | None:
        return self._cerberus_bin_path
