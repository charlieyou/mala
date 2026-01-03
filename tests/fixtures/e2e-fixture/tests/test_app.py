import pytest

from app import add


@pytest.mark.unit
def test_add() -> None:
    assert add(2, 2) == 4
