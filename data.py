"""Dataset loading.

Expected CSV format:
  - a `text` column (article body, or headline if that's all you have)
  - optional `title` column, prepended to `text`
  - a `label` column: 1/0, or fake/real (true/false also accepted)

Label 1 = fake, 0 = real.
"""
from pathlib import Path

import pandas as pd

from fakenews.preprocessing import clean_text

_LABEL_MAP = {"real": 0, "true": 0, "0": 0, "fake": 1, "false": 1, "1": 1}


def _normalize_label(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return int(value) if value in (0, 1) else None
    return _LABEL_MAP.get(str(value).strip().lower())


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    if "title" in df.columns:
        df["text"] = df["title"].fillna("").astype(str) + ". " + df["text"].fillna("").astype(str)

    df["label"] = df["label"].map(_normalize_label)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"])
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]].reset_index(drop=True)
