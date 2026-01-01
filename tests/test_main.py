import runpy
import sys

import pytest


def test_main_help_exits_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["mala", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("src.cli.main", run_name="__main__")

    assert excinfo.value.code in (0, None)
