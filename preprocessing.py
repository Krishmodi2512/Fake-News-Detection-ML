"""Input cleaning and validation."""
import re
import unicodedata

from fakenews import config

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def clean_text(text, max_chars: int = config.MAX_CHARS) -> str:
    """Normalize unicode, replace URLs, collapse whitespace, and truncate."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub(" URL ", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text[:max_chars]


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def validate_input(raw_text) -> tuple[str, list[str]]:
    """Clean user input and return (cleaned_text, warnings).

    Raises ValueError if the input is empty after cleaning.
    """
    warnings: list[str] = []
    original_length = len(str(raw_text)) if raw_text is not None else 0
    cleaned = clean_text(raw_text)

    if not cleaned:
        raise ValueError("Input is empty. Please enter a headline or article text.")

    if original_length > config.MAX_CHARS:
        warnings.append(
            f"Input was truncated to {config.MAX_CHARS:,} characters for analysis."
        )

    if word_count(cleaned) < config.MIN_WORDS:
        warnings.append(
            f"Very short input ({word_count(cleaned)} words). "
            "Predictions on short text are less reliable."
        )

    return cleaned, warnings
