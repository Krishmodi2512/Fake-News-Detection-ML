import pytest

from fakenews.preprocessing import clean_text, validate_input


def test_clean_text_collapses_whitespace_and_replaces_urls():
    out = clean_text("Hello   world\n\nsee https://example.com/x now")
    assert out == "Hello world see URL now"


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_clean_text_truncates():
    assert len(clean_text("a" * 50, max_chars=10)) == 10


def test_validate_rejects_empty():
    with pytest.raises(ValueError):
        validate_input("   ")


def test_validate_warns_on_short_input():
    _, warnings = validate_input("Breaking news today")
    assert any("short" in w.lower() for w in warnings)


def test_validate_no_warning_for_normal_input():
    _, warnings = validate_input("The city council approved the new transit budget on Thursday.")
    assert warnings == []
