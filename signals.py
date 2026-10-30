"""Interpretable style signals shown alongside the model prediction.

These are reported to the user as analysis. They are not used as model inputs,
so they explain writing style rather than drive the prediction.
"""
import re

SENSATIONAL_TERMS = [
    "shocking", "unbelievable", "you won't believe", "bombshell", "explosive",
    "miracle", "secret", "exposed", "cover-up", "hoax", "banned", "deleted",
    "they don't want you", "wake up", "share before", "hate", "destroy",
    "doctors hate", "insiders", "leaked", "breaking",
]

SOURCE_RE = re.compile(
    r"\b(according to|reported|said|says|told reporters|spokesperson|spokespeople|"
    r"official|officials|statement|press release|filing|study|researchers|"
    r"published in|data from|documents show)\b",
    re.IGNORECASE,
)


def analyze_style(text: str) -> dict:
    """Return style metrics and human-readable flags for a cleaned text."""
    words = re.findall(r"[A-Za-z']+", text)
    word_total = max(len(words), 1)
    letters = [c for c in text if c.isalpha()]

    upper_letter_ratio = sum(c.isupper() for c in letters) / max(len(letters), 1)
    all_caps_words = [w for w in words if len(w) > 1 and w.isupper()]
    all_caps_ratio = len(all_caps_words) / word_total
    exclamations = text.count("!")
    questions = text.count("?")
    lowered = text.lower()
    sensational_hits = sorted({t for t in SENSATIONAL_TERMS if t in lowered})
    has_source_language = bool(SOURCE_RE.search(text))

    flags: list[str] = []
    if all_caps_ratio > 0.15:
        flags.append(f"High share of ALL-CAPS words ({all_caps_ratio:.0%}).")
    if exclamations >= 2:
        flags.append(f"Multiple exclamation marks ({exclamations}).")
    if sensational_hits:
        flags.append("Sensational phrasing: " + ", ".join(sensational_hits) + ".")
    if not has_source_language:
        flags.append("No clear source attribution language found.")
    else:
        flags.append("Contains source attribution language (e.g. 'according to', 'officials said').")

    return {
        "word_count": len(words),
        "upper_letter_ratio": round(upper_letter_ratio, 3),
        "all_caps_ratio": round(all_caps_ratio, 3),
        "exclamation_count": exclamations,
        "question_count": questions,
        "sensational_terms": sensational_hits,
        "has_source_language": has_source_language,
        "flags": flags,
    }
