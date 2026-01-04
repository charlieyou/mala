"""Unit tests for coverage argument rewriting helpers."""

from __future__ import annotations


from src.domain.validation.coverage_args import (
    extract_marker_expr,
    normalize_marker_expr,
    rewrite_coverage_command,
    strip_cov_fail_under,
    strip_cov_report_xml,
    strip_xdist_flags,
)


class TestStripXdistFlags:
    """Tests for strip_xdist_flags."""

    def test_strips_dash_n_with_value(self) -> None:
        args = ["pytest", "-n", "4", "tests/"]
        result = strip_xdist_flags(args)
        assert result == ["pytest", "tests/"]

    def test_strips_dash_n_equals(self) -> None:
        args = ["pytest", "-n=auto", "tests/"]
        result = strip_xdist_flags(args)
        assert result == ["pytest", "tests/"]

    def test_strips_numprocesses_with_value(self) -> None:
        args = ["pytest", "--numprocesses", "8", "tests/"]
        result = strip_xdist_flags(args)
        assert result == ["pytest", "tests/"]

    def test_strips_numprocesses_equals(self) -> None:
        args = ["pytest", "--numprocesses=8", "tests/"]
        result = strip_xdist_flags(args)
        assert result == ["pytest", "tests/"]

    def test_strips_dist_with_value(self) -> None:
        args = ["pytest", "--dist", "loadscope", "tests/"]
        result = strip_xdist_flags(args)
        assert result == ["pytest", "tests/"]

    def test_strips_dist_equals(self) -> None:
        args = ["pytest", "--dist=load", "tests/"]
        result = strip_xdist_flags(args)
        assert result == ["pytest", "tests/"]

    def test_preserves_other_args(self) -> None:
        args = ["pytest", "-v", "--cov=src", "tests/"]
        result = strip_xdist_flags(args)
        assert result == ["pytest", "-v", "--cov=src", "tests/"]

    def test_empty_list(self) -> None:
        assert strip_xdist_flags([]) == []


class TestStripCovFailUnder:
    """Tests for strip_cov_fail_under."""

    def test_strips_equals_form(self) -> None:
        args = ["pytest", "--cov-fail-under=80", "tests/"]
        result = strip_cov_fail_under(args)
        assert result == ["pytest", "tests/"]

    def test_strips_separate_value(self) -> None:
        args = ["pytest", "--cov-fail-under", "90", "tests/"]
        result = strip_cov_fail_under(args)
        assert result == ["pytest", "tests/"]

    def test_preserves_other_args(self) -> None:
        args = ["pytest", "--cov=src", "-v", "tests/"]
        result = strip_cov_fail_under(args)
        assert result == ["pytest", "--cov=src", "-v", "tests/"]

    def test_empty_list(self) -> None:
        assert strip_cov_fail_under([]) == []


class TestExtractMarkerExpr:
    """Tests for extract_marker_expr."""

    def test_extracts_separate_value(self) -> None:
        args = ["pytest", "-m", "unit", "tests/"]
        result_args, marker = extract_marker_expr(args)
        assert result_args == ["pytest", "tests/"]
        assert marker == "unit"

    def test_extracts_equals_form(self) -> None:
        args = ["pytest", "-m=integration", "tests/"]
        result_args, marker = extract_marker_expr(args)
        assert result_args == ["pytest", "tests/"]
        assert marker == "integration"

    def test_no_marker_returns_none(self) -> None:
        args = ["pytest", "-v", "tests/"]
        result_args, marker = extract_marker_expr(args)
        assert result_args == ["pytest", "-v", "tests/"]
        assert marker is None

    def test_complex_marker_expression(self) -> None:
        args = ["pytest", "-m", "unit or integration", "tests/"]
        result_args, marker = extract_marker_expr(args)
        assert result_args == ["pytest", "tests/"]
        assert marker == "unit or integration"

    def test_empty_list(self) -> None:
        result_args, marker = extract_marker_expr([])
        assert result_args == []
        assert marker is None


class TestNormalizeMarkerExpr:
    """Tests for normalize_marker_expr."""

    def test_none_returns_default(self) -> None:
        assert normalize_marker_expr(None) == "unit or integration"

    def test_strips_whitespace(self) -> None:
        assert normalize_marker_expr("  unit  ") == "unit"

    def test_e2e_replaced_with_default(self) -> None:
        assert normalize_marker_expr("e2e") == "unit or integration"

    def test_e2e_in_expression_replaced(self) -> None:
        assert normalize_marker_expr("unit or e2e") == "unit or integration"

    def test_preserves_valid_markers(self) -> None:
        assert normalize_marker_expr("unit") == "unit"
        assert normalize_marker_expr("integration") == "integration"
        assert normalize_marker_expr("unit or integration") == "unit or integration"


class TestStripCovReportXml:
    """Tests for strip_cov_report_xml."""

    def test_strips_bare_xml(self) -> None:
        args = ["pytest", "--cov-report=xml", "tests/"]
        result = strip_cov_report_xml(args)
        assert result == ["pytest", "tests/"]

    def test_strips_xml_with_path(self) -> None:
        args = ["pytest", "--cov-report=xml:coverage.xml", "tests/"]
        result = strip_cov_report_xml(args)
        assert result == ["pytest", "tests/"]

    def test_preserves_other_cov_report(self) -> None:
        args = ["pytest", "--cov-report=html", "--cov-report=term", "tests/"]
        result = strip_cov_report_xml(args)
        assert result == ["pytest", "--cov-report=html", "--cov-report=term", "tests/"]

    def test_empty_list(self) -> None:
        assert strip_cov_report_xml([]) == []


class TestRewriteCoverageCommand:
    """Tests for rewrite_coverage_command."""

    def test_basic_command(self) -> None:
        cmd = "pytest --cov=src tests/"
        result = rewrite_coverage_command(cmd, "coverage.xml")
        assert result == [
            "pytest",
            "--cov=src",
            "tests/",
            "--cov-report=xml:coverage.xml",
            "--cov-fail-under=0",
            "-m",
            "unit or integration",
        ]

    def test_strips_xdist_and_fail_under(self) -> None:
        cmd = "pytest -n auto --cov-fail-under=80 --cov=src tests/"
        result = rewrite_coverage_command(cmd, "out.xml")
        assert "-n" not in result
        assert "auto" not in result
        assert "--cov-fail-under=80" not in result
        assert "--cov-fail-under=0" in result

    def test_extracts_and_normalizes_marker(self) -> None:
        cmd = "pytest -m unit --cov=src tests/"
        result = rewrite_coverage_command(cmd, "coverage.xml")
        # Marker should be extracted and re-added at end
        assert result[-2:] == ["-m", "unit"]

    def test_e2e_marker_normalized(self) -> None:
        cmd = "pytest -m e2e --cov=src tests/"
        result = rewrite_coverage_command(cmd, "coverage.xml")
        assert result[-2:] == ["-m", "unit or integration"]

    def test_replaces_existing_xml_report(self) -> None:
        cmd = "pytest --cov-report=xml:old.xml --cov=src tests/"
        result = rewrite_coverage_command(cmd, "new.xml")
        assert "--cov-report=xml:old.xml" not in result
        assert "--cov-report=xml:new.xml" in result

    def test_complex_command(self) -> None:
        cmd = (
            "uv run pytest -n 4 --dist=loadscope "
            "--cov=src --cov-report=xml:coverage.xml --cov-fail-under=85 "
            "-m 'unit or e2e' tests/"
        )
        result = rewrite_coverage_command(cmd, "baseline.xml")
        # xdist stripped
        assert "-n" not in result
        assert "4" not in result
        assert "--dist=loadscope" not in result
        # Original xml stripped, new added
        assert "--cov-report=xml:coverage.xml" not in result
        assert "--cov-report=xml:baseline.xml" in result
        # fail-under reset
        assert "--cov-fail-under=85" not in result
        assert "--cov-fail-under=0" in result
        # e2e normalized
        assert result[-2:] == ["-m", "unit or integration"]
