"""Schema validation tests for the `coder` and `coder_options.amp.mode` fields.

These tests own AC#13 from the AMP provider plan
(plans/2026-04-29-amp-provider-plan.md L194-L215):

> Invalid values fail validation (`src/domain/validation/config.py`) **before**
> any agent process starts.

The data model and resolver precedence (CLI > env > yaml > default) live in
`src/infra/io/config.py` (T002). This module's only job is to fail fast on
invalid YAML.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.domain.validation.config import ConfigError, ValidationConfig
from src.domain.validation.config_loader import load_config

if TYPE_CHECKING:
    from pathlib import Path


class TestCoderEnum:
    """Strict-enum validation for the top-level `coder` field."""

    def test_claude_is_valid(self) -> None:
        ValidationConfig.from_dict({"coder": "claude"})

    def test_amp_is_valid(self) -> None:
        ValidationConfig.from_dict({"coder": "amp"})

    def test_unknown_value_rejected(self) -> None:
        with pytest.raises(
            ConfigError, match=r"coder must be 'claude' or 'amp', got 'foo'"
        ):
            ValidationConfig.from_dict({"coder": "foo"})

    def test_non_string_rejected(self) -> None:
        with pytest.raises(ConfigError, match=r"coder must be 'claude' or 'amp'"):
            ValidationConfig.from_dict({"coder": 42})

    def test_null_is_treated_as_unset(self) -> None:
        # Explicit null should not raise — equivalent to omitting the field.
        ValidationConfig.from_dict({"coder": None})


class TestAmpMode:
    """Strict-enum validation for `coder_options.amp.mode`."""

    @pytest.mark.parametrize("mode", ["smart", "rush", "deep"])
    def test_valid_modes(self, mode: str) -> None:
        ValidationConfig.from_dict({"coder_options": {"amp": {"mode": mode}}})

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(
            ConfigError,
            match=r"coder_options\.amp\.mode must be 'smart', 'rush', or 'deep', got 'bar'",
        ):
            ValidationConfig.from_dict({"coder_options": {"amp": {"mode": "bar"}}})

    def test_non_string_mode_rejected(self) -> None:
        with pytest.raises(ConfigError, match=r"coder_options\.amp\.mode must be"):
            ValidationConfig.from_dict({"coder_options": {"amp": {"mode": 7}}})

    def test_coder_options_not_object_rejected(self) -> None:
        with pytest.raises(
            ConfigError, match=r"coder_options must be an object, got str"
        ):
            ValidationConfig.from_dict({"coder_options": "amp"})

    def test_amp_not_object_rejected(self) -> None:
        with pytest.raises(
            ConfigError, match=r"coder_options\.amp must be an object, got list"
        ):
            ValidationConfig.from_dict({"coder_options": {"amp": ["mode", "smart"]}})


class TestOptionalDefaults:
    """Existing mala.yaml without coder fields must still validate (regression)."""

    def test_omitting_both_fields_is_valid(self) -> None:
        # Sanity: existing yaml shape (no coder, no coder_options) parses cleanly.
        cfg = ValidationConfig.from_dict({"preset": "python-uv"})
        assert cfg.preset == "python-uv"

    def test_coder_options_without_amp_subkey_is_valid(self) -> None:
        # T002's resolver applies amp defaults downstream when amp is omitted.
        ValidationConfig.from_dict({"coder_options": {}})

    def test_amp_without_mode_subkey_is_valid(self) -> None:
        ValidationConfig.from_dict({"coder_options": {"amp": {}}})

    def test_coder_options_null_is_valid(self) -> None:
        ValidationConfig.from_dict({"coder_options": None})

    def test_amp_mode_null_is_valid(self) -> None:
        ValidationConfig.from_dict({"coder_options": {"amp": {"mode": None}}})


class TestEndToEndViaLoadConfig:
    """Verify the top-level allow-list lets these fields through `load_config`."""

    def _write_yaml(self, tmp_path: Path, body: str) -> Path:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "mala.yaml").write_text(body, encoding="utf-8")
        return repo

    def test_coder_field_accepted_at_top_level(self, tmp_path: Path) -> None:
        repo = self._write_yaml(
            tmp_path,
            "preset: python-uv\ncoder: amp\n",
        )
        # Must not raise "Unknown field 'coder'" — the allow-list permits it.
        load_config(repo)

    def test_coder_options_field_accepted_at_top_level(self, tmp_path: Path) -> None:
        repo = self._write_yaml(
            tmp_path,
            "preset: python-uv\ncoder_options:\n  amp:\n    mode: deep\n",
        )
        load_config(repo)

    def test_invalid_coder_rejected_via_load_config(self, tmp_path: Path) -> None:
        repo = self._write_yaml(
            tmp_path,
            "preset: python-uv\ncoder: foo\n",
        )
        with pytest.raises(ConfigError, match=r"coder must be 'claude' or 'amp'"):
            load_config(repo)

    def test_invalid_amp_mode_rejected_via_load_config(self, tmp_path: Path) -> None:
        repo = self._write_yaml(
            tmp_path,
            "preset: python-uv\ncoder_options:\n  amp:\n    mode: bar\n",
        )
        with pytest.raises(ConfigError, match=r"coder_options\.amp\.mode must be"):
            load_config(repo)
