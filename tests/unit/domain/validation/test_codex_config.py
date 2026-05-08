"""Schema validation tests for ``coder_options.codex.*`` (Phase B, AC #13).

The yaml shape is:

    coder: codex
    coder_options:
      codex:
        model: gpt-5.5
        effort: high
        approval_policy: never
        sandbox: danger-full-access

Strict-enum validation lives in
:mod:`src.domain.validation.config`. Resolver precedence is exercised in
:mod:`tests.unit.cli.test_codex_flags` and the orchestration factory tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.domain.validation.config import (
    CodexOptionsConfig,
    ConfigError,
    ValidationConfig,
)
from src.domain.validation.config_loader import load_config

if TYPE_CHECKING:
    from pathlib import Path


def _yaml(tmp_path: Path, body: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "mala.yaml").write_text(body, encoding="utf-8")
    return repo


# ---------------------------------------------------------------------------
# Defaults / unset
# ---------------------------------------------------------------------------


class TestUnset:
    def test_omitting_codex_options_is_valid(self) -> None:
        cfg = ValidationConfig.from_dict({"coder": "codex"})
        assert cfg.codex_options is None

    def test_empty_codex_options_block_yields_default_optional_config(self) -> None:
        cfg = ValidationConfig.from_dict({"coder_options": {"codex": {}}})
        assert cfg.codex_options == CodexOptionsConfig()

    def test_null_codex_block_yields_default_optional_config(self) -> None:
        cfg = ValidationConfig.from_dict({"coder_options": {"codex": None}})
        assert cfg.codex_options == CodexOptionsConfig()


# ---------------------------------------------------------------------------
# Top-level coder_options shape
# ---------------------------------------------------------------------------


class TestCoderOptionsShape:
    def test_unknown_top_level_subkey_rejected(self) -> None:
        """Only ``codex`` is allowed under ``coder_options``."""
        with pytest.raises(ConfigError, match=r"Unknown field 'amp' in coder_options"):
            ValidationConfig.from_dict({"coder_options": {"amp": {"mode": "deep"}}})

    def test_non_object_coder_options_rejected(self) -> None:
        with pytest.raises(ConfigError, match=r"coder_options must be an object"):
            ValidationConfig.from_dict({"coder_options": "codex"})


# ---------------------------------------------------------------------------
# Codex sub-block validation
# ---------------------------------------------------------------------------


class TestCodexBlock:
    def test_unknown_codex_field_rejected(self) -> None:
        with pytest.raises(
            ConfigError, match=r"Unknown field 'foo' in coder_options.codex"
        ):
            ValidationConfig.from_dict({"coder_options": {"codex": {"foo": "bar"}}})

    @pytest.mark.parametrize(
        "policy", ["never", "on-request", "on-failure", "untrusted"]
    )
    def test_valid_approval_policies(self, policy: str) -> None:
        cfg = ValidationConfig.from_dict(
            {"coder_options": {"codex": {"approval_policy": policy}}}
        )
        assert cfg.codex_options is not None
        assert cfg.codex_options.approval_policy == policy

    def test_invalid_approval_policy_rejected(self) -> None:
        with pytest.raises(
            ConfigError,
            match=r"coder_options.codex.approval_policy must be one of .*never.*",
        ):
            ValidationConfig.from_dict(
                {"coder_options": {"codex": {"approval_policy": "always"}}}
            )

    @pytest.mark.parametrize(
        "sandbox", ["read-only", "workspace-write", "danger-full-access"]
    )
    def test_valid_sandboxes(self, sandbox: str) -> None:
        cfg = ValidationConfig.from_dict(
            {"coder_options": {"codex": {"sandbox": sandbox}}}
        )
        assert cfg.codex_options is not None
        assert cfg.codex_options.sandbox == sandbox

    def test_invalid_sandbox_rejected(self) -> None:
        with pytest.raises(
            ConfigError, match=r"coder_options.codex.sandbox must be one of"
        ):
            ValidationConfig.from_dict(
                {"coder_options": {"codex": {"sandbox": "yolo"}}}
            )

    def test_invalid_effort_rejected(self) -> None:
        with pytest.raises(ConfigError, match=r"Invalid Codex effort 'super-high'"):
            ValidationConfig.from_dict(
                {"coder_options": {"codex": {"effort": "super-high"}}}
            )

    def test_empty_model_rejected(self) -> None:
        with pytest.raises(ConfigError, match=r"non-empty string"):
            ValidationConfig.from_dict({"coder_options": {"codex": {"model": ""}}})

    def test_full_block_round_trips(self) -> None:
        cfg = ValidationConfig.from_dict(
            {
                "coder": "codex",
                "coder_options": {
                    "codex": {
                        "model": "gpt-5.5-foo",
                        "approval_policy": "never",
                        "sandbox": "danger-full-access",
                    }
                },
            }
        )
        assert cfg.coder == "codex"
        assert cfg.codex_options is not None
        assert cfg.codex_options.model == "gpt-5.5-foo"
        assert cfg.codex_options.approval_policy == "never"
        assert cfg.codex_options.sandbox == "danger-full-access"


# ---------------------------------------------------------------------------
# End-to-end via load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_codex_options_accepted_at_top_level(self, tmp_path: Path) -> None:
        repo = _yaml(
            tmp_path,
            "preset: python-uv\ncoder: codex\n"
            "coder_options:\n"
            "  codex:\n"
            "    model: gpt-5.5-foo\n"
            "    approval_policy: never\n"
            "    sandbox: danger-full-access\n",
        )
        cfg = load_config(repo)
        assert cfg.coder == "codex"
        assert cfg.codex_options is not None
        assert cfg.codex_options.model == "gpt-5.5-foo"

    def test_invalid_sandbox_rejected_via_load_config(self, tmp_path: Path) -> None:
        repo = _yaml(
            tmp_path,
            "preset: python-uv\ncoder: codex\n"
            "coder_options:\n"
            "  codex:\n"
            "    sandbox: yolo\n",
        )
        with pytest.raises(ConfigError, match=r"sandbox must be one of"):
            load_config(repo)

    def test_invalid_approval_policy_rejected_via_load_config(
        self, tmp_path: Path
    ) -> None:
        repo = _yaml(
            tmp_path,
            "preset: python-uv\ncoder: codex\n"
            "coder_options:\n"
            "  codex:\n"
            "    approval_policy: always\n",
        )
        with pytest.raises(ConfigError, match=r"approval_policy must be one of"):
            load_config(repo)

    def test_invalid_effort_rejected_via_load_config(self, tmp_path: Path) -> None:
        repo = _yaml(
            tmp_path,
            "preset: python-uv\ncoder: codex\n"
            "coder_options:\n"
            "  codex:\n"
            "    effort: super-high\n",
        )
        with pytest.raises(ConfigError, match=r"Invalid Codex effort"):
            load_config(repo)
