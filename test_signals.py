from fakenews.signals import analyze_style


def test_detects_all_caps_and_exclamations():
    result = analyze_style("SHOCKING NEWS!!! THE GOVERNMENT LIED!!")
    assert result["all_caps_ratio"] > 0.15
    assert result["exclamation_count"] == 5
    assert any("ALL-CAPS" in f for f in result["flags"])


def test_detects_sensational_terms():
    result = analyze_style("Doctors hate this secret miracle cure")
    assert "doctors hate" in result["sensational_terms"]
    assert "secret" in result["sensational_terms"]


def test_detects_source_attribution():
    result = analyze_style("The agency said on Monday that prices rose, according to officials.")
    assert result["has_source_language"] is True


def test_flags_missing_source():
    result = analyze_style("Something happened and you should share it with everyone.")
    assert result["has_source_language"] is False
    assert any("No clear source" in f for f in result["flags"])
