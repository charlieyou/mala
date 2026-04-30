"""Unit tests for :class:`AmpPluginInstaller`.

Covers the installer slice of plan section "Testing & Validation"
(``plans/2026-04-29-amp-provider-plan.md#L805-L811``):

- First call writes ``~/.config/amp/plugins/mala-safety.ts``.
- Second call is a no-op when content hash matches.
- When content differs, the file is replaced via temp-then-rename.
- Concurrent calls don't corrupt the file.
- Acknowledgment header is preserved verbatim in the installed copy.
- Missing or unwritable plugin directory fails with a clear error.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import pytest

from src.infra.clients.amp_plugin_installer import (
    DEFAULT_PLUGIN_DIR,
    PLUGIN_FILENAME,
    AmpPluginInstaller,
    AmpPluginInstallError,
    InstallResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_REAL_BUNDLED_PLUGIN = (
    Path(__file__).resolve().parents[4] / "plugins" / "amp" / PLUGIN_FILENAME
)


@pytest.fixture
def fake_source(tmp_path: Path) -> Path:
    """A fake bundled plugin file with predictable content."""
    src = tmp_path / "src-plugin" / PLUGIN_FILENAME
    src.parent.mkdir(parents=True)
    src.write_bytes(b"// fake mala-safety plugin v1\n// body\n")
    return src


@pytest.fixture
def target_dir(tmp_path: Path) -> Path:
    """A clean target directory that does not yet exist."""
    return tmp_path / "amp-plugins"


@pytest.fixture
def installer(fake_source: Path) -> AmpPluginInstaller:
    return AmpPluginInstaller(source_path=fake_source)


# ---------------------------------------------------------------------------
# Constructor / source resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_source_resolves_to_bundled_plugin() -> None:
    inst = AmpPluginInstaller()
    assert inst.source_path == _REAL_BUNDLED_PLUGIN


@pytest.mark.unit
def test_default_target_directory_is_amp_global() -> None:
    assert DEFAULT_PLUGIN_DIR == Path("~/.config/amp/plugins").expanduser()


@pytest.mark.unit
def test_explicit_source_path_is_used(fake_source: Path) -> None:
    inst = AmpPluginInstaller(source_path=fake_source)
    assert inst.source_path == fake_source


# ---------------------------------------------------------------------------
# Plan L807: first call writes the file with the bundled bytes.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_first_call_creates_target_dir_and_writes_file(
    installer: AmpPluginInstaller, fake_source: Path, target_dir: Path
) -> None:
    assert not target_dir.exists()

    result = installer.install(target_dir=target_dir)

    assert isinstance(result, InstallResult)
    assert result.action == "wrote"
    assert result.target_path == target_dir / PLUGIN_FILENAME
    assert result.target_path.is_file()
    assert result.target_path.read_bytes() == fake_source.read_bytes()


@pytest.mark.unit
def test_first_call_uses_mkdir_p_through_missing_parents(
    fake_source: Path, tmp_path: Path
) -> None:
    deeply_nested = tmp_path / "a" / "b" / "c" / "amp-plugins"
    inst = AmpPluginInstaller(source_path=fake_source)

    result = inst.install(target_dir=deeply_nested)

    assert result.target_path == deeply_nested / PLUGIN_FILENAME
    assert result.target_path.is_file()


@pytest.mark.unit
def test_install_returns_truncated_sha256_version_hash(
    installer: AmpPluginInstaller, fake_source: Path, target_dir: Path
) -> None:
    expected_full = hashlib.sha256(fake_source.read_bytes()).hexdigest()
    expected_prefix = expected_full[:16]

    result = installer.install(target_dir=target_dir)

    assert result.plugin_hash == expected_prefix
    assert installer.installed_plugin_hash() == expected_prefix


# ---------------------------------------------------------------------------
# Plan L808: second call with matching hash is a no-op.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_second_call_with_matching_hash_skips_write(
    installer: AmpPluginInstaller, target_dir: Path
) -> None:
    first = installer.install(target_dir=target_dir)
    target_path = first.target_path
    mtime_after_first = target_path.stat().st_mtime_ns
    inode_after_first = target_path.stat().st_ino

    second = installer.install(target_dir=target_dir)

    assert second.action == "skipped"
    assert second.target_path == target_path
    assert second.plugin_hash == first.plugin_hash
    assert target_path.stat().st_mtime_ns == mtime_after_first
    assert target_path.stat().st_ino == inode_after_first


# ---------------------------------------------------------------------------
# Plan L809: replace stale on-disk content via temp-then-rename.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stale_content_is_replaced_via_atomic_rename(
    installer: AmpPluginInstaller, fake_source: Path, target_dir: Path
) -> None:
    target_dir.mkdir(parents=True)
    stale = target_dir / PLUGIN_FILENAME
    stale.write_bytes(b"// stale plugin from a previous mala version\n")
    stale_inode = stale.stat().st_ino

    result = installer.install(target_dir=target_dir)

    assert result.action == "replaced"
    assert stale.read_bytes() == fake_source.read_bytes()
    # write-temp-then-rename produces a fresh inode (atomic replace).
    assert stale.stat().st_ino != stale_inode


@pytest.mark.unit
def test_no_temp_file_remains_after_replace(
    installer: AmpPluginInstaller, fake_source: Path, target_dir: Path
) -> None:
    del fake_source
    installer.install(target_dir=target_dir)
    installer.install(target_dir=target_dir)

    leftovers = [p.name for p in target_dir.iterdir() if p.name != PLUGIN_FILENAME]
    assert leftovers == []


# ---------------------------------------------------------------------------
# Plan L810: concurrent calls don't corrupt the file.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_concurrent_install_threads_produce_one_uncorrupted_file(
    fake_source: Path, target_dir: Path
) -> None:
    expected_bytes = fake_source.read_bytes()
    expected_hash = hashlib.sha256(expected_bytes).hexdigest()
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []
    results: list[InstallResult] = []
    lock = threading.Lock()

    def worker() -> None:
        inst = AmpPluginInstaller(source_path=fake_source)
        try:
            barrier.wait(timeout=10)
            res = inst.install(target_dir=target_dir)
        except BaseException as exc:
            with lock:
                errors.append(exc)
            return
        with lock:
            results.append(res)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    assert len(results) == 8
    target_path = target_dir / PLUGIN_FILENAME
    assert target_path.is_file()
    assert hashlib.sha256(target_path.read_bytes()).hexdigest() == expected_hash
    assert target_path.read_bytes() == expected_bytes
    leftovers = [p.name for p in target_dir.iterdir() if p.name != PLUGIN_FILENAME]
    assert leftovers == [], f"Stray temp files left: {leftovers}"
    for res in results:
        assert res.action in {"wrote", "skipped", "replaced"}
        assert res.target_path == target_path


# ---------------------------------------------------------------------------
# Plan L811: acknowledgment header preserved verbatim in installed copy.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_acknowledgment_header_preserved_in_installed_copy(target_dir: Path) -> None:
    # Use the *real* bundled plugin so this test catches header-rewrites.
    if not _REAL_BUNDLED_PLUGIN.is_file():
        pytest.skip("Bundled plugin not present in this checkout")

    inst = AmpPluginInstaller()
    result = inst.install(target_dir=target_dir)

    head = result.target_path.read_text(encoding="utf-8").splitlines()[0]
    assert (
        head == "// @i-know-the-amp-plugin-api-is-wip-and-very-experimental-right-now"
    )
    assert result.target_path.read_bytes() == _REAL_BUNDLED_PLUGIN.read_bytes()


# ---------------------------------------------------------------------------
# Plan L812: missing or unwritable plugin directory fails with a clear error.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unwritable_target_directory_raises_clear_error(
    installer: AmpPluginInstaller, tmp_path: Path
) -> None:
    target = tmp_path / "ro-plugins"
    target.mkdir()
    # Read-only directory: prevents temp file creation and rename.
    target.chmod(0o555)
    try:
        with pytest.raises(AmpPluginInstallError) as excinfo:
            installer.install(target_dir=target)
        msg = str(excinfo.value)
        assert "plugin" in msg.lower()
        assert str(target) in msg or PLUGIN_FILENAME in msg
    finally:
        target.chmod(0o755)


@pytest.mark.unit
def test_target_dir_path_blocked_by_existing_file_raises(
    installer: AmpPluginInstaller, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"i am a file, not a dir")

    with pytest.raises(AmpPluginInstallError) as excinfo:
        installer.install(target_dir=blocker)
    assert (
        "plugin directory" in str(excinfo.value).lower()
        or "plugin" in str(excinfo.value).lower()
    )


@pytest.mark.unit
def test_missing_bundled_source_raises_clear_error(
    tmp_path: Path, target_dir: Path
) -> None:
    missing_source = tmp_path / "does-not-exist" / PLUGIN_FILENAME
    inst = AmpPluginInstaller(source_path=missing_source)

    with pytest.raises(AmpPluginInstallError) as excinfo:
        inst.install(target_dir=target_dir)
    msg = str(excinfo.value)
    assert "missing" in msg.lower() or "incomplete" in msg.lower()
    assert str(missing_source) in msg


# ---------------------------------------------------------------------------
# installed_plugin_hash() handles missing source the same way as install().
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_installed_plugin_hash_raises_when_source_missing(tmp_path: Path) -> None:
    missing_source = tmp_path / "no-source" / PLUGIN_FILENAME
    inst = AmpPluginInstaller(source_path=missing_source)
    with pytest.raises(AmpPluginInstallError):
        inst.installed_plugin_hash()


# ---------------------------------------------------------------------------
# Skip-when-already-installed guard does not race with hash computation.
# A skipped install must not modify the target file mtime (regression for
# accidental "rewrite-then-skip" implementations).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_existing_file_with_different_content_triggers_replace(
    fake_source: Path, target_dir: Path
) -> None:
    target_dir.mkdir(parents=True)
    target_path = target_dir / PLUGIN_FILENAME
    target_path.write_bytes(b"// old content\n")

    inst = AmpPluginInstaller(source_path=fake_source)
    result = inst.install(target_dir=target_dir)

    assert result.action == "replaced"
    assert target_path.read_bytes() == fake_source.read_bytes()


@pytest.mark.unit
def test_skipped_install_does_not_touch_inode_or_mtime(
    fake_source: Path, target_dir: Path
) -> None:
    inst = AmpPluginInstaller(source_path=fake_source)
    first = inst.install(target_dir=target_dir)
    stat_before = first.target_path.stat()

    # Force a different system clock tick before the second call so any
    # accidental rewrite would be visible in mtime.
    os.utime(first.target_path, ns=(stat_before.st_atime_ns, 1_000_000_000))

    second = inst.install(target_dir=target_dir)

    assert second.action == "skipped"
    stat_after = second.target_path.stat()
    assert stat_after.st_ino == stat_before.st_ino
    assert stat_after.st_mtime_ns == 1_000_000_000
