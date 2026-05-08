"""Unit tests for :class:`CodexPluginInstaller` (T015 / Phase E4).

Mirrors the slice :mod:`tests.unit.infra.clients.test_amp_plugin_installer`
covers for Amp; the Codex installer differs in that it copies a
*directory tree* (``.codex-plugin/{plugin.json, hooks.json, .mcp.json}``)
rather than a single file. The contract per
``plans/2026-05-07-codex-provider-plan.md#L862-L902`` is otherwise
identical: idempotent install, atomic replace of stale content,
concurrent safety, and a stable ``trusted_hash`` derived from the
bundled bytes.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from src.infra.clients.codex_plugin_installer import (
    PLUGIN_DIRNAME,
    PLUGIN_FILENAMES,
    PLUGIN_NAME,
    CodexPluginInstaller,
    CodexPluginInstallError,
    InstallResult,
    _bundled_source_dir,
    _combined_hash,
    _read_source_files,
    default_plugin_target_dir,
)


_REAL_BUNDLED_DIR = (
    Path(__file__).resolve().parents[4]
    / "plugins"
    / "codex"
    / PLUGIN_NAME
    / PLUGIN_DIRNAME
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_source(tmp_path: Path) -> Path:
    """A fake bundled plugin tree with predictable content."""
    src = tmp_path / "src-plugin"
    src.mkdir()
    (src / "plugin.json").write_bytes(b'{"name": "mala-safety", "version": "0.0.1"}\n')
    (src / "hooks.json").write_bytes(b'{"hooks": {"PreToolUse": []}}\n')
    (src / ".mcp.json").write_bytes(b'{"mcpServers": {}}\n')
    return src


@pytest.fixture
def target_dir(tmp_path: Path) -> Path:
    """A clean target directory that does not yet exist."""
    return tmp_path / "codex-plugin-target"


@pytest.fixture
def installer(fake_source: Path) -> CodexPluginInstaller:
    return CodexPluginInstaller(source_dir=fake_source)


# ---------------------------------------------------------------------------
# Constructor / source resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_source_dir_resolves_to_an_existing_tree() -> None:
    """The default ``source_dir`` must point to a real directory in the
    active install layout — repo plugins/ in source checkout, or
    wheel-data sibling once force-include ships the bundle."""
    inst = CodexPluginInstaller()
    assert inst.source_dir.is_dir(), (
        f"Bundled plugin tree not findable at {inst.source_dir}; "
        "the wheel force-include in pyproject.toml may have regressed "
        "or the source tree is incomplete."
    )
    for name in PLUGIN_FILENAMES:
        assert (inst.source_dir / name).is_file(), (
            f"Bundled plugin file {name} missing under {inst.source_dir}."
        )


@pytest.mark.unit
def test_default_source_resolves_to_repo_plugin_in_source_checkout() -> None:
    """In a source checkout, the resolver returns the repo-plugin tree."""
    inst = CodexPluginInstaller()
    assert inst.source_dir == _REAL_BUNDLED_DIR


@pytest.mark.unit
def test_bundled_manifest_points_at_dot_codex_plugin_paths() -> None:
    """Regression: ``plugin.json`` must reference paths under
    ``./.codex-plugin/`` so Codex resolves them to the actual install
    location.

    Codex's ``resolve_manifest_path``
    (``codex-rs/core-plugins/src/manifest.rs:397``) joins the manifest
    string against the plugin root (the directory CONTAINING
    ``.codex-plugin/``). A bare ``./hooks.json`` would resolve to
    ``<plugin_root>/hooks.json``, which we never install, so the hook
    + MCP would silently not load while the manifest itself parses
    fine.
    """
    import json as _json

    manifest_path = _REAL_BUNDLED_DIR / "plugin.json"
    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["hooks"] == "./.codex-plugin/hooks.json", (
        "plugin.json hooks path must include the .codex-plugin/ prefix; "
        "Codex resolves manifest paths relative to the plugin root, not "
        "the .codex-plugin/ directory itself."
    )
    assert manifest["mcpServers"] == "./.codex-plugin/.mcp.json", (
        "plugin.json mcpServers path must include the .codex-plugin/ prefix."
    )


@pytest.mark.unit
def test_resolver_prefers_wheel_data_dir_when_present(tmp_path: Path) -> None:
    """Simulate a wheel-installed layout: the installer module sits inside
    ``site-packages/src/infra/clients/`` and a ``_codex_plugin_data/``
    sibling holds the shipped tree. The resolver must pick that path,
    not the nonexistent ``parents[3]/plugins/codex/...`` fallback."""
    site_packages = tmp_path / "site-packages"
    module_dir = site_packages / "src" / "infra" / "clients"
    module_dir.mkdir(parents=True)
    fake_module_file = module_dir / "codex_plugin_installer.py"
    fake_module_file.write_bytes(b"# stub for path resolution\n")
    wheel_data_dir = module_dir / "_codex_plugin_data"
    wheel_data_dir.mkdir()
    (wheel_data_dir / "plugin.json").write_bytes(b'{"name":"shipped"}\n')
    (wheel_data_dir / "hooks.json").write_bytes(b"{}\n")
    (wheel_data_dir / ".mcp.json").write_bytes(b"{}\n")

    resolved = _bundled_source_dir(module_file=fake_module_file)

    assert resolved == wheel_data_dir
    assert (resolved / "plugin.json").is_file()


@pytest.mark.unit
def test_default_target_directory_honors_codex_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``default_plugin_target_dir`` reads ``CODEX_HOME`` at call time so
    a redirected env var lands in ``tmp_path``."""
    fake_home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(fake_home))
    expected = fake_home / "plugins" / PLUGIN_NAME / PLUGIN_DIRNAME
    assert default_plugin_target_dir() == expected


@pytest.mark.unit
def test_default_target_directory_falls_back_to_user_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``CODEX_HOME`` → ``~/.codex/plugins/mala-safety/.codex-plugin``."""
    monkeypatch.delenv("CODEX_HOME", raising=False)
    expected = Path("~/.codex").expanduser() / "plugins" / PLUGIN_NAME / PLUGIN_DIRNAME
    assert default_plugin_target_dir() == expected


# ---------------------------------------------------------------------------
# install: first-call write semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_first_call_creates_target_dir_and_writes_files(
    installer: CodexPluginInstaller, fake_source: Path, target_dir: Path
) -> None:
    assert not target_dir.exists()

    result = installer.install(target_dir=target_dir)

    assert isinstance(result, InstallResult)
    assert result.action == "wrote"
    assert result.target_dir == target_dir
    for name in PLUGIN_FILENAMES:
        assert (target_dir / name).is_file()
        assert (target_dir / name).read_bytes() == (fake_source / name).read_bytes()


@pytest.mark.unit
def test_first_call_uses_mkdir_p_through_missing_parents(
    fake_source: Path, tmp_path: Path
) -> None:
    deeply_nested = tmp_path / "a" / "b" / "c" / "codex-plugin"
    inst = CodexPluginInstaller(source_dir=fake_source)

    result = inst.install(target_dir=deeply_nested)

    assert result.target_dir == deeply_nested
    for name in PLUGIN_FILENAMES:
        assert (deeply_nested / name).is_file()


@pytest.mark.unit
def test_install_returns_truncated_sha256_trusted_hash(
    installer: CodexPluginInstaller, fake_source: Path, target_dir: Path
) -> None:
    payload = _read_source_files(fake_source)
    expected_full = _combined_hash(payload)
    expected_prefix = expected_full[:16]

    result = installer.install(target_dir=target_dir)

    assert result.plugin_hash == expected_prefix
    assert installer.installed_plugin_hash() == expected_prefix


# ---------------------------------------------------------------------------
# install: idempotent on identical content (E4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_second_call_with_matching_hash_skips_write(
    installer: CodexPluginInstaller, target_dir: Path
) -> None:
    first = installer.install(target_dir=target_dir)
    target_path = first.target_dir / "plugin.json"
    mtime_after_first = target_path.stat().st_mtime_ns
    inode_after_first = target_path.stat().st_ino

    second = installer.install(target_dir=target_dir)

    assert second.action == "skipped"
    assert second.target_dir == first.target_dir
    assert second.plugin_hash == first.plugin_hash
    assert target_path.stat().st_mtime_ns == mtime_after_first
    assert target_path.stat().st_ino == inode_after_first


@pytest.mark.unit
def test_skipped_install_does_not_touch_inode_or_mtime(
    fake_source: Path, target_dir: Path
) -> None:
    """A skipped install must not modify any file's mtime or inode (E4)."""
    inst = CodexPluginInstaller(source_dir=fake_source)
    first = inst.install(target_dir=target_dir)
    plugin_path = first.target_dir / "plugin.json"
    stat_before = plugin_path.stat()

    # Force a clock tick before the second call so any accidental rewrite
    # would be visible in mtime.
    os.utime(plugin_path, ns=(stat_before.st_atime_ns, 1_000_000_000))

    second = inst.install(target_dir=target_dir)

    assert second.action == "skipped"
    stat_after = plugin_path.stat()
    assert stat_after.st_ino == stat_before.st_ino
    assert stat_after.st_mtime_ns == 1_000_000_000


# ---------------------------------------------------------------------------
# install: stale content is replaced via temp-then-rename (atomic)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stale_content_is_replaced_via_atomic_rename(
    installer: CodexPluginInstaller, fake_source: Path, target_dir: Path
) -> None:
    target_dir.mkdir(parents=True)
    for name in PLUGIN_FILENAMES:
        (target_dir / name).write_bytes(b"// stale\n")
    stale_inode = (target_dir / "plugin.json").stat().st_ino

    result = installer.install(target_dir=target_dir)

    assert result.action == "replaced"
    for name in PLUGIN_FILENAMES:
        assert (target_dir / name).read_bytes() == (fake_source / name).read_bytes()
    # write-temp-then-rename produces a fresh inode (atomic replace).
    assert (target_dir / "plugin.json").stat().st_ino != stale_inode


@pytest.mark.unit
def test_no_temp_file_remains_after_replace(
    installer: CodexPluginInstaller, fake_source: Path, target_dir: Path
) -> None:
    del fake_source
    installer.install(target_dir=target_dir)
    installer.install(target_dir=target_dir)

    leftovers = sorted(
        p.name for p in target_dir.iterdir() if p.name not in PLUGIN_FILENAMES
    )
    assert leftovers == []


@pytest.mark.unit
def test_partial_install_is_treated_as_replace(
    installer: CodexPluginInstaller, fake_source: Path, target_dir: Path
) -> None:
    """If only a subset of files exists (e.g. a previous install crashed
    midway), the next install must rewrite all files — not silently skip."""
    target_dir.mkdir(parents=True)
    (target_dir / "plugin.json").write_bytes((fake_source / "plugin.json").read_bytes())
    # hooks.json + .mcp.json missing.

    result = installer.install(target_dir=target_dir)

    assert result.action == "wrote"
    for name in PLUGIN_FILENAMES:
        assert (target_dir / name).is_file()
        assert (target_dir / name).read_bytes() == (fake_source / name).read_bytes()


@pytest.mark.unit
def test_no_temp_file_leaks_when_fsync_fails(
    installer: CodexPluginInstaller,
    target_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An I/O error during ``write/flush/fsync`` must still trigger
    cleanup so partial temp files don't accumulate (parity with the
    Amp installer's ENOSPC regression).
    """
    import src.infra.clients.codex_plugin_installer as installer_mod

    def failing_fsync(fd: int) -> None:
        del fd
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(installer_mod.os, "fsync", failing_fsync)

    target_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(CodexPluginInstallError):
        installer.install(target_dir=target_dir)

    leftovers = list(target_dir.iterdir())
    assert leftovers == [], (
        f"Failed install left stray temp files behind: {[p.name for p in leftovers]}."
    )


# ---------------------------------------------------------------------------
# Concurrent install safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_concurrent_install_threads_produce_one_uncorrupted_tree(
    fake_source: Path, target_dir: Path
) -> None:
    expected_payload = _read_source_files(fake_source)
    expected_hash = _combined_hash(expected_payload)
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []
    results: list[InstallResult] = []
    lock = threading.Lock()

    def worker() -> None:
        inst = CodexPluginInstaller(source_dir=fake_source)
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
    on_disk_payload = _read_source_files(target_dir)
    assert _combined_hash(on_disk_payload) == expected_hash
    leftovers = sorted(
        p.name for p in target_dir.iterdir() if p.name not in PLUGIN_FILENAMES
    )
    assert leftovers == [], f"Stray temp files left: {leftovers}"
    for res in results:
        assert res.action in {"wrote", "skipped", "replaced"}
        assert res.target_dir == target_dir


# ---------------------------------------------------------------------------
# Error paths: unwritable directory, blocking file, missing source
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unwritable_target_directory_raises_clear_error(
    installer: CodexPluginInstaller, tmp_path: Path
) -> None:
    target = tmp_path / "ro-plugin"
    target.mkdir()
    target.chmod(0o555)  # Read-only: can't create temp files inside.
    try:
        with pytest.raises(CodexPluginInstallError) as excinfo:
            installer.install(target_dir=target)
        msg = str(excinfo.value)
        assert "plugin" in msg.lower()
        assert str(target) in msg or "Codex" in msg
    finally:
        target.chmod(0o755)


@pytest.mark.unit
def test_target_dir_path_blocked_by_existing_file_raises(
    installer: CodexPluginInstaller, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"i am a file, not a dir")

    with pytest.raises(CodexPluginInstallError):
        installer.install(target_dir=blocker)


@pytest.mark.unit
def test_missing_bundled_source_raises_clear_error(
    tmp_path: Path, target_dir: Path
) -> None:
    missing_source = tmp_path / "does-not-exist"
    inst = CodexPluginInstaller(source_dir=missing_source)

    with pytest.raises(CodexPluginInstallError) as excinfo:
        inst.install(target_dir=target_dir)
    msg = str(excinfo.value)
    assert "missing" in msg.lower() or "incomplete" in msg.lower()


@pytest.mark.unit
def test_installed_plugin_hash_raises_when_source_missing(tmp_path: Path) -> None:
    inst = CodexPluginInstaller(source_dir=tmp_path / "no-source")
    with pytest.raises(CodexPluginInstallError):
        inst.installed_plugin_hash()


# ---------------------------------------------------------------------------
# Hash determinism: bundled source and on-disk install agree
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_install_hash_stable_against_dict_iteration_order(
    fake_source: Path, target_dir: Path
) -> None:
    """The trusted-hash must depend only on bytes, not on filesystem
    iteration order — files are read by the fixed ``PLUGIN_FILENAMES``
    sequence so two runs on different filesystems produce the same hash.
    """
    inst = CodexPluginInstaller(source_dir=fake_source)
    first = inst.install(target_dir=target_dir).plugin_hash

    # Re-read source bytes manually and confirm the combined hash is the
    # truncated-sha256 of the deterministic length-prefixed payload.
    payload = _read_source_files(fake_source)
    assert _combined_hash(payload)[:16] == first
