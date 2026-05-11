"""Unit tests for :mod:`src.infra.clients.codex_selftest` pure helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING, cast

import pytest

from src.infra.clients.codex_selftest import (
    _build_module_hash_probe_code,
    _canonical_json,
    _compute_normalized_hook_hash,
    _normalized_hook_identity_value,
)

if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.unit


class TestNormalizedHookIdentityValue:
    def test_default_event_payload_matches_codex_serde_shape(self) -> None:
        value = _normalized_hook_identity_value()

        assert value == {
            "event_name": "pre_tool_use",
            "hooks": [
                {
                    "type": "command",
                    "command": "mala-codex-pre-tool-use",
                    "timeout": 600,
                    "async": False,
                }
            ],
        }

    @pytest.mark.parametrize(
        "event",
        ["pre_tool_use", "session_start"],
    )
    def test_event_name_is_propagated_unchanged(self, event: str) -> None:
        value = _normalized_hook_identity_value(event)

        assert isinstance(value, dict)
        assert cast(dict[str, object], value)["event_name"] == event


class TestCanonicalJson:
    def test_sorts_dict_keys_recursively(self) -> None:
        assert _canonical_json({"b": 1, "a": {"d": 2, "c": 3}}) == {
            "a": {"c": 3, "d": 2},
            "b": 1,
        }

    def test_preserves_list_order(self) -> None:
        assert _canonical_json([3, 1, 2]) == [3, 1, 2]

    def test_normalizes_dicts_inside_lists(self) -> None:
        assert _canonical_json([{"b": 1, "a": 2}]) == [{"a": 2, "b": 1}]

    @pytest.mark.parametrize(
        "scalar",
        [1, "x", True, False, None, 1.5],
    )
    def test_passes_scalars_through(self, scalar: object) -> None:
        assert _canonical_json(scalar) == scalar


class TestComputeNormalizedHookHash:
    def test_hash_is_deterministic_sha256_hex_with_prefix(self) -> None:
        digest = _compute_normalized_hook_hash()

        assert digest.startswith("sha256:")
        hex_part = digest.split(":", 1)[1]
        assert len(hex_part) == 64
        int(hex_part, 16)  # valid hex

    def test_matches_canonical_json_sha256_of_identity_payload(self) -> None:
        payload = _canonical_json(_normalized_hook_identity_value("pre_tool_use"))
        serialized = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        expected = f"sha256:{hashlib.sha256(serialized).hexdigest()}"

        assert _compute_normalized_hook_hash("pre_tool_use") == expected

    def test_distinct_events_produce_distinct_hashes(self) -> None:
        assert _compute_normalized_hook_hash(
            "pre_tool_use"
        ) != _compute_normalized_hook_hash("session_start")


class TestBuildModuleHashProbeCode:
    def test_embeds_module_tuple_literal(self) -> None:
        modules = ("a.b", "c.d")

        code = _build_module_hash_probe_code(modules)

        assert "modules = ('a.b', 'c.d')" in code
        assert code.endswith("print(h.hexdigest())\n")

    def test_probe_emits_combined_hash_matching_in_process_computation(
        self, tmp_path: Path
    ) -> None:
        # Two on-disk modules the probe will hash via ``find_spec``.
        pkg = tmp_path / "probe_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        mod_a = pkg / "mod_a.py"
        mod_a.write_text("ALPHA = 1\n", encoding="utf-8")
        mod_b = pkg / "mod_b.py"
        mod_b.write_text("BETA = 'two'\n", encoding="utf-8")

        modules = ("probe_pkg.mod_a", "probe_pkg.mod_b")

        # Expected digest, computed using the same length-prefix scheme
        # the probe code emits.
        expected = hashlib.sha256()
        for name, path in (
            (modules[0], mod_a),
            (modules[1], mod_b),
        ):
            data = path.read_bytes()
            expected.update(name.encode("utf-8"))
            expected.update(b"\0")
            expected.update(len(data).to_bytes(8, "big"))
            expected.update(data)

        probe_code = _build_module_hash_probe_code(modules)
        script = tmp_path / "probe.py"
        script.write_text(probe_code, encoding="utf-8")

        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

        assert completed.stdout.strip() == expected.hexdigest()

    def test_probe_reports_nomodule_and_exits_nonzero_when_missing(
        self, tmp_path: Path
    ) -> None:
        probe_code = _build_module_hash_probe_code(("definitely_no_such_module",))
        script = tmp_path / "probe.py"
        script.write_text(probe_code, encoding="utf-8")

        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 2
        assert completed.stdout.strip() == "NOMODULE:definitely_no_such_module"

    def test_probe_code_compiles(self) -> None:
        code = _build_module_hash_probe_code(("os", "sys"))
        # Sanity check: the emitted source must be syntactically valid
        # Python so the on-PATH interpreter can execute it.
        compile(textwrap.dedent(code), "<probe>", "exec")
