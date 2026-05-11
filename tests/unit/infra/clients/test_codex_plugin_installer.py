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
    PLUGIN_MARKETPLACE,
    PLUGIN_NAME,
    PLUGIN_VERSION,
    CodexPluginInstaller,
    CodexPluginInstallError,
    InstallResult,
    _bundled_source_dir,
    _combined_hash,
    _ensure_key_in_section,
    _hook_state_key,
    _plugin_id,
    _read_source_files,
    _resolve_codex_home,
    _rewrite_toml_block,
    _write_codex_plugin_config,
    default_plugin_target_dir,
    plugin_root_dir,
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
def test_bundled_mcp_forwards_lock_context_env_vars() -> None:
    """Regression: Codex stdio MCP subprocesses get a sanitized env.

    The static bundled ``.mcp.json`` must list Mala's lock-context
    variables in ``env_vars`` so Codex forwards them from the
    app-server process env to ``mala-codex-mcp-locking``. An empty
    ``env`` map alone is not enough.
    """
    import json as _json

    payload = _json.loads((_REAL_BUNDLED_DIR / ".mcp.json").read_text("utf-8"))
    bundled = payload["mcpServers"]["mala-locking"]
    assert bundled["env_vars"] == [
        "MALA_AGENT_ID",
        "MALA_LOCK_DIR",
        "MALA_REPO_NAMESPACE",
    ]


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
def test_default_target_directory_lands_in_codex_plugin_store_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: the install target must match Codex's PluginStore cache
    layout (``$CODEX_HOME/plugins/cache/<marketplace>/<plugin>/<version>/``)
    plus the ``.codex-plugin/`` manifest sub-directory.

    Codex's loader (``codex-rs/core-plugins/src/store.rs::PluginStore``)
    only resolves plugin roots from this exact path; installing into
    ``$CODEX_HOME/plugins/<plugin>/.codex-plugin/`` (the pre-fix
    location) would leave the bundled hook invisible to discovery —
    Codex's ``active_plugin_root`` would return None even though the
    files are physically on disk.
    """
    fake_home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(fake_home))
    expected = (
        fake_home
        / "plugins"
        / "cache"
        / PLUGIN_MARKETPLACE
        / PLUGIN_NAME
        / PLUGIN_VERSION
        / PLUGIN_DIRNAME
    )
    assert default_plugin_target_dir() == expected


@pytest.mark.unit
def test_plugin_root_dir_matches_plugin_store_active_root_layout(
    tmp_path: Path,
) -> None:
    """Regression-pin for the cache-path contract Codex's
    ``PluginStore.plugin_root`` enforces."""
    home = tmp_path / "codex-home"
    expected = home / "plugins" / "cache" / "local" / "mala-safety" / "local"
    assert plugin_root_dir(home) == expected


@pytest.mark.unit
def test_default_target_directory_falls_back_to_user_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``CODEX_HOME`` → ``~/.codex/plugins/cache/local/mala-safety/local/.codex-plugin``."""
    monkeypatch.delenv("CODEX_HOME", raising=False)
    expected = (
        Path("~/.codex").expanduser()
        / "plugins"
        / "cache"
        / PLUGIN_MARKETPLACE
        / PLUGIN_NAME
        / PLUGIN_VERSION
        / PLUGIN_DIRNAME
    )
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
def test_default_install_lands_at_plugin_store_active_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: a default ``install()`` call must land where Codex's
    ``PluginStore.active_plugin_root`` would resolve to, so the bundled
    plugin appears in ``configured_plugins_from_stack`` and its
    PreToolUse hook fires.

    The pre-fix install path
    (``$CODEX_HOME/plugins/<plugin>/.codex-plugin/``) put the manifest
    on disk but outside Codex's PluginStore cache; ``active_plugin_root``
    returned None and the hook was never invoked. This test exercises
    the real bundled source so the cross-test contract — manifest
    bytes + filesystem path — both match the install Codex actually
    discovers.
    """
    fake_home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(fake_home))

    inst = CodexPluginInstaller()
    result = inst.install()  # default target dir

    expected_root = (
        fake_home
        / "plugins"
        / "cache"
        / PLUGIN_MARKETPLACE
        / PLUGIN_NAME
        / PLUGIN_VERSION
    )
    expected_dir = expected_root / PLUGIN_DIRNAME
    assert result.target_dir == expected_dir
    assert (expected_dir / "plugin.json").is_file()
    assert (expected_dir / "hooks.json").is_file()
    assert (expected_dir / ".mcp.json").is_file()
    # PluginStore.active_plugin_version reads version dirs under
    # plugin_base_root and prefers ``local`` when present — verifies
    # the plugin is at the path Codex's discovery would land on.
    assert plugin_root_dir(fake_home) == expected_root


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


# ---------------------------------------------------------------------------
# TOML config rewriting helpers (relocated from codex_provider.py, T_B3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_codex_home_honors_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_HOME", "/tmp/fake-codex-home")
    assert _resolve_codex_home() == Path("/tmp/fake-codex-home")


@pytest.mark.unit
def test_resolve_codex_home_falls_back_to_user_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEX_HOME", raising=False)
    assert _resolve_codex_home() == Path("~/.codex").expanduser()


@pytest.mark.unit
def test_plugin_id_uses_plugin_marketplace_default() -> None:
    assert _plugin_id() == f"{PLUGIN_NAME}@{PLUGIN_MARKETPLACE}"


@pytest.mark.unit
def test_plugin_id_accepts_custom_marketplace() -> None:
    assert _plugin_id("other") == f"{PLUGIN_NAME}@other"


@pytest.mark.unit
def test_hook_state_key_default_event_pre_tool_use() -> None:
    """Key shape mirrors ``codex-rs/hooks/src/declarations.rs::plugin_hook_key_source``."""
    expected = (
        f"{PLUGIN_NAME}@{PLUGIN_MARKETPLACE}:.codex-plugin/hooks.json:pre_tool_use:0:0"
    )
    assert _hook_state_key() == expected


@pytest.mark.unit
def test_hook_state_key_session_start_event() -> None:
    expected = (
        f"{PLUGIN_NAME}@{PLUGIN_MARKETPLACE}:.codex-plugin/hooks.json:session_start:0:0"
    )
    assert _hook_state_key("session_start") == expected


@pytest.mark.unit
def test_ensure_key_in_section_preserves_well_formed_toml_without_trailing_newline() -> (
    None
):
    """Regression: ``_ensure_key_in_section`` must not concatenate the
    new key onto the previous line when the input lacks a trailing
    newline.

    A user-edited ``config.toml`` ending mid-section without a final
    ``\\n`` would previously round-trip to malformed TOML
    (``plugins = trueplugin_hooks = true``) which crashes Codex on
    startup.
    """
    existing_no_trailing_newline = "[features]\nplugins = true"  # no '\n' at EOF

    rewritten = _ensure_key_in_section(
        existing_no_trailing_newline,
        section_header="[features]",
        key="plugin_hooks",
        value="true",
    )

    # The output must contain the existing key on its own line and the
    # new key on its own line — no concatenation.
    lines = rewritten.splitlines()
    assert "plugins = true" in lines
    assert "plugin_hooks = true" in lines
    # And the merged token ``trueplugin_hooks`` must NOT appear.
    assert "trueplugin_hooks" not in rewritten


@pytest.mark.unit
def test_ensure_key_in_section_handles_aligned_spaces_around_equals() -> None:
    """Regression: TOML allows arbitrary whitespace between key and ``=``
    (e.g. ``plugins    = false`` for value alignment). The matcher must
    treat such lines as the same key — otherwise the rewriter appends a
    second ``plugins = true`` line, producing a duplicate-key TOML file
    that Codex rejects on startup.
    """
    aligned = "[features]\nplugins    = false\nplugin_hooks    = false\n"

    rewritten = _ensure_key_in_section(
        aligned, section_header="[features]", key="plugins", value="true"
    )

    # ``plugins = false`` must be flipped to ``plugins = true``; the
    # original aligned-spaces line is rewritten in place.
    assert "plugins = true" in rewritten
    assert "plugins    = false" not in rewritten
    # No duplicate ``plugins`` entries — the in-place rewrite did not
    # also append a fresh ``plugins = true`` line.
    plugin_lines = [
        ln
        for ln in rewritten.splitlines()
        if ln.strip().startswith("plugins")
        and not ln.strip().startswith("plugin_hooks")
    ]
    assert len(plugin_lines) == 1, plugin_lines


@pytest.mark.unit
def test_ensure_key_in_section_does_not_match_key_prefix() -> None:
    """Regression: looking for ``plugin`` must not match
    ``plugin_hooks`` — otherwise the rewriter overwrites the wrong
    line. Pins the key-bound check ``stripped[len(key):].lstrip().startswith("=")``
    behavior.
    """
    # Only ``plugin_hooks`` exists; we ask for ``plugin`` (a true prefix
    # but a distinct key). The rewriter must NOT rewrite the
    # ``plugin_hooks`` line — it should leave that line alone and
    # append a fresh ``plugin = true`` instead.
    existing = "[features]\nplugin_hooks = true\n"

    rewritten = _ensure_key_in_section(
        existing, section_header="[features]", key="plugin", value="true"
    )

    assert "plugin_hooks = true" in rewritten  # untouched
    # And the new ``plugin = true`` line was added inside the section.
    plugin_lines = [
        ln for ln in rewritten.splitlines() if ln.strip() == "plugin = true"
    ]
    assert len(plugin_lines) == 1, rewritten


@pytest.mark.unit
def test_rewrite_toml_block_replaces_existing_section() -> None:
    existing = (
        '[plugins."mala-safety@local"]\nenabled = false\n\n[other]\nfoo = "bar"\n'
    )
    new_block = '[plugins."mala-safety@local"]\nenabled = true\n'

    rewritten = _rewrite_toml_block(
        existing,
        section_header='[plugins."mala-safety@local"]',
        new_block=new_block,
    )

    assert "enabled = true" in rewritten
    assert "enabled = false" not in rewritten
    # Sibling sections are preserved.
    assert "[other]" in rewritten
    assert 'foo = "bar"' in rewritten


@pytest.mark.unit
def test_rewrite_toml_block_appends_when_missing() -> None:
    rewritten = _rewrite_toml_block(
        '[other]\nfoo = "bar"\n',
        section_header='[plugins."mala-safety@local"]',
        new_block='[plugins."mala-safety@local"]\nenabled = true\n',
    )
    assert '[plugins."mala-safety@local"]' in rewritten
    assert "enabled = true" in rewritten
    assert "[other]" in rewritten


@pytest.mark.unit
def test_write_codex_plugin_config_writes_all_five_preconditions(
    tmp_path: Path,
) -> None:
    """All five Codex-config preconditions land in the rewritten file."""
    codex_home = tmp_path / "codex-home"
    _write_codex_plugin_config(codex_home=codex_home)

    rendered = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert "[features]" in rendered
    assert "plugins = true" in rendered
    assert "plugin_hooks = true" in rendered
    assert "hooks = true" in rendered
    assert '[plugins."mala-safety@local"]' in rendered
    # Per-hook trust blocks for both events.
    assert (
        '[hooks.state."mala-safety@local:.codex-plugin/hooks.json:pre_tool_use:0:0"]'
        in rendered
    )
    assert (
        '[hooks.state."mala-safety@local:.codex-plugin/hooks.json:session_start:0:0"]'
        in rendered
    )
    assert "trusted_hash" in rendered


@pytest.mark.unit
def test_write_codex_plugin_config_reaches_fixed_point(tmp_path: Path) -> None:
    """Rewrites stabilize: once the file content matches the target shape,
    the next call short-circuits without touching the file (no mtime change).

    The first call may collapse interstitial blank lines as it splices in
    each block; once collapsed, subsequent calls see ``rewritten ==
    existing`` and return without writing.
    """
    codex_home = tmp_path / "codex-home"
    _write_codex_plugin_config(codex_home=codex_home)
    # Drive to fixed point.
    _write_codex_plugin_config(codex_home=codex_home)
    stable_text = (codex_home / "config.toml").read_text(encoding="utf-8")
    # Force a clock tick before the no-op call so any accidental rewrite
    # would be visible in mtime.
    os.utime(codex_home / "config.toml", ns=(1_000_000_000, 1_000_000_000))

    _write_codex_plugin_config(codex_home=codex_home)

    assert (codex_home / "config.toml").read_text(encoding="utf-8") == stable_text
    assert (codex_home / "config.toml").stat().st_mtime_ns == 1_000_000_000


@pytest.mark.unit
def test_write_codex_plugin_config_preserves_unrelated_features(
    tmp_path: Path,
) -> None:
    """A pre-existing ``[features]`` key is preserved across the rewrite."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        "[features]\nremote_plugin = true\n", encoding="utf-8"
    )

    _write_codex_plugin_config(codex_home=codex_home)

    rendered = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert "remote_plugin = true" in rendered
    assert "plugins = true" in rendered
    assert "plugin_hooks = true" in rendered
    assert "hooks = true" in rendered


@pytest.mark.unit
def test_write_codex_plugin_config_raises_on_unwritable_home(
    tmp_path: Path,
) -> None:
    """An unwritable Codex home raises ``CodexHookNotActiveError`` so the
    caller fails closed instead of proceeding under
    ``danger-full-access`` without the safety gate."""
    from src.infra.clients.codex_provider import (
        CodexHookNotActiveError,
        CodexHookNotActiveReason,
    )

    parent = tmp_path / "ro-parent"
    parent.mkdir()
    parent.chmod(0o555)
    target = parent / "codex-home"
    try:
        with pytest.raises(CodexHookNotActiveError) as excinfo:
            _write_codex_plugin_config(codex_home=target)
    finally:
        parent.chmod(0o755)
    assert excinfo.value.reason == CodexHookNotActiveReason.TRUSTED_HASH_MISMATCH
