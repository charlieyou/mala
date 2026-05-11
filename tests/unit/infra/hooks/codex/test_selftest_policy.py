"""Focused unit tests for ``src.infra.hooks.codex.selftest_policy``.

Pins every branch of the extracted selftest marker / live-probe policy
so a regression here is caught without rerunning the whole hook adapter.
Adapter-level wiring (``main()`` → marker emit → event routing) is
exercised by the golden corpus harness and the existing
``test_codex_pre_tool_use.py`` integration tests; this module covers the
pure-policy surface directly.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest

from src.infra.hooks.codex.selftest_policy import (
    SELFTEST_IDENTITY_MODULES,
    SELFTEST_MARKER_DIR_ENV,
    SELFTEST_MARKER_ENV,
    SELFTEST_TARGET_EVENT_ENV,
    _atomic_write_json,
    _compute_selftest_identity_hash,
    _emit_selftest_marker_if_requested,
    _handle_pre_tool_use_selftest,
    _handle_session_start_event,
    _selftest_mode_active,
    _selftest_target_event_matches,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Env-var names (pin the literal contracts duplicated in codex_provider)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnvVarLiterals:
    """The provider duplicates these strings; drift breaks the live probe."""

    def test_marker_env_literal(self) -> None:
        assert SELFTEST_MARKER_ENV == "MALA_CODEX_HOOK_SELFTEST_MARKER"

    def test_marker_dir_env_literal(self) -> None:
        assert SELFTEST_MARKER_DIR_ENV == "MALA_CODEX_HOOK_SELFTEST_MARKER_DIR"

    def test_target_event_env_literal(self) -> None:
        assert SELFTEST_TARGET_EVENT_ENV == "MALA_CODEX_HOOK_SELFTEST_TARGET_EVENT"


# ---------------------------------------------------------------------------
# Identity hash computation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIdentityHash:
    def test_includes_codex_pre_tool_use_and_selftest_policy(self) -> None:
        """Both the entry-point and this module must be in the identity set.

        Self-extraction safety: divergence in ``selftest_policy.py``
        (e.g. a stale install with skewed marker payload bytes) must
        flip the hash, so the module that owns the identity tuple
        must itself be in the tuple.
        """
        assert "src.infra.hooks.codex_pre_tool_use" in SELFTEST_IDENTITY_MODULES
        assert "src.infra.hooks.codex.selftest_policy" in SELFTEST_IDENTITY_MODULES

    def test_returns_hex_digest_of_expected_length(self) -> None:
        result = _compute_selftest_identity_hash()
        assert result is not None
        assert len(result) == 64  # SHA-256 hex length
        int(result, 16)  # raises if non-hex

    def test_is_stable_across_calls(self) -> None:
        """Same module bytes → identical digest. Pins the contract the
        provider's parallel computation relies on."""
        assert _compute_selftest_identity_hash() == _compute_selftest_identity_hash()


# ---------------------------------------------------------------------------
# Atomic marker write
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAtomicWriteJson:
    def test_writes_json_payload(self, tmp_path: Path) -> None:
        target = tmp_path / "marker.json"
        _atomic_write_json(target, {"x": 1, "y": "z"})
        assert json.loads(target.read_text(encoding="utf-8")) == {"x": 1, "y": "z"}

    def test_uses_sibling_temp_via_os_replace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: the marker write must go through a sibling temp +
        ``os.replace`` so a polling reader never sees a partial / empty
        file (review-4 P1 TOCTOU). A direct ``Path.write_text`` would
        expose ``open + write`` non-atomicity to a parallel reader."""
        target = tmp_path / "marker.json"
        calls: list[tuple[str, str]] = []
        real_replace = os.replace

        def tracking_replace(src: object, dst: object) -> None:
            calls.append((str(src), str(dst)))
            real_replace(str(src), str(dst))

        monkeypatch.setattr(os, "replace", tracking_replace)
        _atomic_write_json(target, {"ok": True})

        assert target.is_file()
        matching = [src for src, dst in calls if dst == str(target)]
        assert matching, f"_atomic_write_json did not go through os.replace: {calls!r}"
        # Source path lives next to the target so the rename is atomic.
        from pathlib import Path as _Path

        assert _Path(matching[0]).parent == target.parent


# ---------------------------------------------------------------------------
# Marker emit branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmitMarker:
    def test_no_envs_set_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(SELFTEST_MARKER_ENV, raising=False)
        monkeypatch.delenv(SELFTEST_MARKER_DIR_ENV, raising=False)
        _emit_selftest_marker_if_requested("PreToolUse")
        assert list(tmp_path.iterdir()) == []

    def test_writes_single_file_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        marker = tmp_path / "marker.json"
        monkeypatch.setenv(SELFTEST_MARKER_ENV, str(marker))
        monkeypatch.delenv(SELFTEST_MARKER_DIR_ENV, raising=False)

        _emit_selftest_marker_if_requested("PreToolUse")

        payload = json.loads(marker.read_text(encoding="utf-8"))
        assert payload["mala_codex_hook"] == "loaded"
        assert payload["version"] == _compute_selftest_identity_hash()
        assert payload["event_name"] == "PreToolUse"

    def test_writes_per_event_marker_in_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        marker_dir = tmp_path / "markers"
        marker_dir.mkdir()
        monkeypatch.delenv(SELFTEST_MARKER_ENV, raising=False)
        monkeypatch.setenv(SELFTEST_MARKER_DIR_ENV, str(marker_dir))

        _emit_selftest_marker_if_requested("SessionStart")

        marker = marker_dir / "SessionStart.json"
        payload = json.loads(marker.read_text(encoding="utf-8"))
        assert payload["event_name"] == "SessionStart"

    def test_per_event_marker_rejects_unsafe_event_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Path-traversal guard: only the known event names produce a
        per-event marker file (a malicious ``hook_event_name`` like
        ``../etc/passwd`` must NOT escape the marker dir)."""
        marker_dir = tmp_path / "markers"
        marker_dir.mkdir()
        monkeypatch.delenv(SELFTEST_MARKER_ENV, raising=False)
        monkeypatch.setenv(SELFTEST_MARKER_DIR_ENV, str(marker_dir))

        _emit_selftest_marker_if_requested("../../escape")

        assert list(marker_dir.iterdir()) == []

    def test_per_event_marker_requires_event_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty event_name disables the per-event marker even if the
        dir env is set (e.g. invalid JSON payload → no hook_event_name)."""
        marker_dir = tmp_path / "markers"
        marker_dir.mkdir()
        monkeypatch.delenv(SELFTEST_MARKER_ENV, raising=False)
        monkeypatch.setenv(SELFTEST_MARKER_DIR_ENV, str(marker_dir))

        _emit_selftest_marker_if_requested("")

        assert list(marker_dir.iterdir()) == []

    def test_writes_both_markers_when_both_envs_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        marker = tmp_path / "marker.json"
        marker_dir = tmp_path / "markers"
        marker_dir.mkdir()
        monkeypatch.setenv(SELFTEST_MARKER_ENV, str(marker))
        monkeypatch.setenv(SELFTEST_MARKER_DIR_ENV, str(marker_dir))

        _emit_selftest_marker_if_requested("PreToolUse")

        assert marker.is_file()
        assert (marker_dir / "PreToolUse.json").is_file()

    def test_swallows_write_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unwritable marker path must not raise — the provider's
        selftest reads the absence of the marker as the fail-closed
        signal (HOOK_MARKER_MISSING)."""
        bad = tmp_path / "missing-dir" / "marker.json"
        monkeypatch.setenv(SELFTEST_MARKER_ENV, str(bad))
        monkeypatch.delenv(SELFTEST_MARKER_DIR_ENV, raising=False)

        _emit_selftest_marker_if_requested("PreToolUse")  # must not raise

        assert not bad.exists()

    def test_swallows_per_event_write_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad_dir = tmp_path / "missing-dir"
        monkeypatch.delenv(SELFTEST_MARKER_ENV, raising=False)
        monkeypatch.setenv(SELFTEST_MARKER_DIR_ENV, str(bad_dir))

        _emit_selftest_marker_if_requested("PreToolUse")  # must not raise

        assert not bad_dir.exists()


# ---------------------------------------------------------------------------
# Mode / target helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelftestModeActive:
    def test_false_when_no_envs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(SELFTEST_MARKER_ENV, raising=False)
        monkeypatch.delenv(SELFTEST_MARKER_DIR_ENV, raising=False)
        assert _selftest_mode_active() is False

    def test_true_when_single_file_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(SELFTEST_MARKER_ENV, "/tmp/marker.json")
        monkeypatch.delenv(SELFTEST_MARKER_DIR_ENV, raising=False)
        assert _selftest_mode_active() is True

    def test_true_when_marker_dir_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(SELFTEST_MARKER_ENV, raising=False)
        monkeypatch.setenv(SELFTEST_MARKER_DIR_ENV, "/tmp/markers")
        assert _selftest_mode_active() is True


@pytest.mark.unit
class TestSelftestTargetEventMatches:
    def test_false_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(SELFTEST_TARGET_EVENT_ENV, raising=False)
        assert _selftest_target_event_matches("PreToolUse") is False

    def test_false_when_env_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SELFTEST_TARGET_EVENT_ENV, "")
        assert _selftest_target_event_matches("PreToolUse") is False

    def test_true_on_exact_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SELFTEST_TARGET_EVENT_ENV, "PreToolUse")
        assert _selftest_target_event_matches("PreToolUse") is True

    def test_false_on_mismatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SELFTEST_TARGET_EVENT_ENV, "SessionStart")
        assert _selftest_target_event_matches("PreToolUse") is False


# ---------------------------------------------------------------------------
# Session-start decision branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandleSessionStart:
    def test_production_returns_continue_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(SELFTEST_MARKER_ENV, raising=False)
        monkeypatch.delenv(SELFTEST_MARKER_DIR_ENV, raising=False)
        monkeypatch.delenv(SELFTEST_TARGET_EVENT_ENV, raising=False)
        assert _handle_session_start_event() == {"continue": True}

    def test_selftest_no_target_aborts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Selftest mode + no target event → ``continue=false`` so the
        SessionStart probe aborts before any LLM call."""
        monkeypatch.setenv(SELFTEST_MARKER_ENV, "/tmp/marker.json")
        monkeypatch.delenv(SELFTEST_MARKER_DIR_ENV, raising=False)
        monkeypatch.delenv(SELFTEST_TARGET_EVENT_ENV, raising=False)

        result = _handle_session_start_event()

        assert result["continue"] is False
        assert "stopReason" in result

    def test_selftest_target_session_start_aborts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(SELFTEST_MARKER_ENV, "/tmp/marker.json")
        monkeypatch.setenv(SELFTEST_TARGET_EVENT_ENV, "SessionStart")

        result = _handle_session_start_event()

        assert result["continue"] is False
        assert "stopReason" in result

    def test_selftest_target_pre_tool_use_continues(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the live PreToolUse probe is driving, SessionStart must
        let the turn proceed (``continue=true``) so Codex reaches the
        model and emits a tool call which triggers PreToolUse."""
        monkeypatch.setenv(SELFTEST_MARKER_DIR_ENV, "/tmp/markers")
        monkeypatch.setenv(SELFTEST_TARGET_EVENT_ENV, "PreToolUse")

        assert _handle_session_start_event() == {"continue": True}


# ---------------------------------------------------------------------------
# Pre-tool-use selftest decision
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandlePreToolUseSelftest:
    def test_returns_block_decision_with_deny_reason(self) -> None:
        result = _handle_pre_tool_use_selftest()

        assert result["decision"] == "block"
        assert "selftest probe" in result["reason"]
        spec = result["hookSpecificOutput"]
        assert spec["hookEventName"] == "PreToolUse"
        assert spec["permissionDecision"] == "deny"
        assert spec["permissionDecisionReason"] == "mala-codex selftest probe"

    def test_omits_universal_continue_fields(self) -> None:
        """Regression: Codex's PreToolUse output parser rejects the
        universal ``continue``/``stopReason`` fields as unsupported, and
        the PreToolUse runner fails open on invalid hook output.
        Emitting them would let the sentinel tool call proceed even
        though the selftest marker was written and the probe appeared
        to pass — silently defeating the deny/abort the live dispatch
        probe relies on."""
        result = _handle_pre_tool_use_selftest()

        assert "continue" not in result
        assert "stopReason" not in result
