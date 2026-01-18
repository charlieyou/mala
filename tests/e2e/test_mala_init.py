"""E2E tests for mala init config generation.

These tests verify:
1. mala init generates valid YAML with evidence_check and validation_triggers sections
2. Generated config is structurally valid for mala run (via dry-run validation)
3. Skip flags correctly omit sections

Note: Full behavioral verification (commands executed, run_end triggered) requires
Claude SDK integration. That is covered by TestE2ERunnerIntegration.test_run_real_fixture
in test_e2e.py, which spawns actual Claude agents and verifies run_validation metadata.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.e2e


def _skip_if_missing_cli() -> None:
    """Skip test if mala or bd CLI is not available."""
    if shutil.which("mala") is None:
        pytest.skip("mala CLI not found in PATH")
    if shutil.which("br") is None:
        pytest.skip("br CLI not found in PATH")


def _init_git_repo(repo_path: Path) -> None:
    """Initialize a git repo with stub config for testing."""
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )


def _create_stub_pyproject(repo_path: Path) -> None:
    """Create a minimal pyproject.toml for python-uv preset testing."""
    pyproject = repo_path / "pyproject.toml"
    pyproject.write_text(
        """\
[project]
name = "test-project"
version = "0.1.0"
requires-python = ">=3.10"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
    )


def _create_stub_python_file(repo_path: Path) -> None:
    """Create a stub Python file so the repo has content."""
    src_dir = repo_path / "src"
    src_dir.mkdir(exist_ok=True)
    (src_dir / "main.py").write_text(
        '"""Stub module for testing."""\n\ndef main() -> None:\n    pass\n'
    )


class TestInitGeneratesCorrectYamlStructure:
    """Tests verifying mala init produces correct YAML structure."""

    def test_init_with_preset_produces_evidence_and_triggers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """mala init --yes --preset python-uv generates evidence_check and validation_triggers."""
        _skip_if_missing_cli()
        monkeypatch.chdir(tmp_path)
        _init_git_repo(tmp_path)
        _create_stub_pyproject(tmp_path)
        _create_stub_python_file(tmp_path)
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        # Run mala init
        result = subprocess.run(
            ["mala", "init", "--yes", "--preset", "python-uv"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"mala init failed: {result.stderr}"

        # Verify mala.yaml was created
        mala_yaml = tmp_path / "mala.yaml"
        assert mala_yaml.exists(), "mala.yaml not created"

        # Parse and verify structure
        config = yaml.safe_load(mala_yaml.read_text())

        # Check preset
        assert config.get("preset") == "python-uv"

        # Check evidence_check section exists and has required commands
        assert "evidence_check" in config, "evidence_check section missing"
        evidence = config["evidence_check"]
        assert "required" in evidence, "evidence_check.required missing"
        required = evidence["required"]
        # python-uv preset defaults to test and lint for evidence
        assert isinstance(required, list)
        assert "test" in required
        assert "lint" in required

        # Check validation_triggers section exists
        assert "validation_triggers" in config, "validation_triggers section missing"
        triggers = config["validation_triggers"]

        # Check run_end trigger exists
        assert "run_end" in triggers, "run_end trigger missing"
        run_end = triggers["run_end"]

        # Check run_end has commands with ref syntax
        assert "commands" in run_end, "run_end.commands missing"
        commands = run_end["commands"]
        assert isinstance(commands, list)
        # Commands should be dicts with 'ref' key
        assert len(commands) > 0, "run_end.commands is empty"
        for cmd in commands:
            assert isinstance(cmd, dict), f"Expected dict, got {type(cmd)}"
            assert "ref" in cmd, f"Command missing 'ref' key: {cmd}"

        # Check failure_mode is set
        assert "failure_mode" in run_end, "run_end.failure_mode missing"

    def test_init_skip_flags_omit_sections(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """mala init --skip-evidence --skip-triggers omits those sections."""
        _skip_if_missing_cli()
        monkeypatch.chdir(tmp_path)
        _init_git_repo(tmp_path)
        _create_stub_pyproject(tmp_path)
        _create_stub_python_file(tmp_path)
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        # Run mala init with skip flags
        result = subprocess.run(
            [
                "mala",
                "init",
                "--yes",
                "--preset",
                "python-uv",
                "--skip-evidence",
                "--skip-triggers",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"mala init failed: {result.stderr}"

        # Parse config
        mala_yaml = tmp_path / "mala.yaml"
        config = yaml.safe_load(mala_yaml.read_text())

        # Sections should be omitted, not empty
        assert "evidence_check" not in config
        assert "validation_triggers" not in config


class TestInitConfigCompatibility:
    """Tests verifying init config is compatible with mala run.

    These tests validate that init-generated config can be loaded by mala run
    without errors (via --dry-run which validates config before processing).

    Note: Actual command execution and run_end trigger firing requires Claude SDK.
    That behavioral verification is in TestE2ERunnerIntegration.test_run_real_fixture
    (test_e2e.py), which uses E2ERunner with auth checks and verifies run_validation.
    """

    def test_init_with_stub_commands_produces_runnable_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify init config allows mala run (via dry-run validation).

        This test:
        1. Creates a fixture repo with stub commands (echo ok)
        2. Runs mala init to generate config
        3. Overrides commands in mala.yaml to use stubs
        4. Creates a mock issue via bd
        5. Runs mala run --dry-run to verify config is valid and issue is picked up

        Full behavioral verification (commands executed, run_end triggered) is
        covered by test_run_real_fixture in test_e2e.py which uses Claude SDK.
        """
        _skip_if_missing_cli()
        monkeypatch.chdir(tmp_path)
        _init_git_repo(tmp_path)
        _create_stub_pyproject(tmp_path)
        _create_stub_python_file(tmp_path)
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        # Run mala init
        result = subprocess.run(
            ["mala", "init", "--yes", "--preset", "python-uv"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"mala init failed: {result.stderr}"

        # Override commands with stubs for deterministic testing
        mala_yaml = tmp_path / "mala.yaml"
        config = yaml.safe_load(mala_yaml.read_text())

        # Replace all commands with echo stubs
        config["commands"] = {
            "test": "echo test-ok",
            "lint": "echo lint-ok",
            "format": "echo format-ok",
            "typecheck": "echo typecheck-ok",
            "setup": "echo setup-ok",
        }

        mala_yaml.write_text(yaml.safe_dump(config, default_flow_style=False))

        # Initialize beads for issue tracking
        result = subprocess.run(
            ["br", "init"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"br init failed: {result.stderr}"

        # Create a test issue
        result = subprocess.run(
            ["br", "create", "Test issue", "-p", "1"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"br create failed: {result.stderr}"

        # Run mala run --dry-run to verify config is valid and issue is picked up
        # Note: cwd=tmp_path means mala run uses current dir (no positional arg needed)
        result = subprocess.run(
            ["mala", "run", "--dry-run"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"mala run --dry-run failed: {result.stderr}"

        # Verify dry-run output shows the task was found
        # dry-run lists tasks in format: "  1. [T-1] Test issue"
        output = result.stdout + result.stderr
        assert "Test issue" in output or "T-1" in output, (
            f"Dry-run output doesn't show the created issue: {output}"
        )
