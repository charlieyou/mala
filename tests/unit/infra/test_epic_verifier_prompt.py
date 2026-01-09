import pytest

from src.infra.epic_verifier import _load_prompt_template


@pytest.mark.unit
def test_epic_verification_prompt_includes_guidelines() -> None:
    template = _load_prompt_template()
    assert "Verification Checks (mandatory)" in template
    assert "Return only the JSON object" in template


@pytest.mark.unit
def test_epic_verification_prompt_formats_cleanly() -> None:
    template = _load_prompt_template()
    rendered = template.format(
        epic_criteria="- Criterion A",
        spec_content="Spec text",
        commit_range="abc..def",
        commit_list="abc123\ndef456",
    )
    assert "- Criterion A" in rendered
    assert "Spec text" in rendered
    assert "abc..def" in rendered
    assert "abc123" in rendered
    assert "{epic_criteria}" not in rendered
    assert "{spec_content}" not in rendered
    assert "{commit_range}" not in rendered
    assert "{commit_list}" not in rendered
