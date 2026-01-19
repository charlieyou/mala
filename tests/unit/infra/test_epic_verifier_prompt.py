import pytest

from src.infra.epic_verifier import _load_prompt_template


@pytest.mark.unit
def test_epic_verification_prompt_includes_guidelines() -> None:
    template = _load_prompt_template()
    assert "Verification Checks (mandatory)" in template
    assert "Respond with valid JSON only" in template


@pytest.mark.unit
def test_epic_verification_prompt_formats_cleanly() -> None:
    template = _load_prompt_template()
    rendered = template.format(
        epic_context="## Epic Description\n\n- Criterion A\n\n## Spec Content\n\nSpec text",
    )
    assert "- Criterion A" in rendered
    assert "Spec text" in rendered
    assert "{epic_context}" not in rendered
