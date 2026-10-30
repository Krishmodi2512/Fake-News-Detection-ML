"""Model: TF-IDF features + logistic regression, with Platt-scaled probabilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from fakenews import config


@dataclass
class FakeNewsModel:
    vectorizer: TfidfVectorizer
    calibrated: CalibratedClassifierCV
    # Uncalibrated model on the same features. Used only for per-term
    # contributions, since its coefficients map directly to individual words.
    explainer: LogisticRegression

    def predict_proba_fake(self, texts: list[str]) -> np.ndarray:
        """Calibrated probability that each text is fake (label 1)."""
        X = self.vectorizer.transform(texts)
        return self.calibrated.predict_proba(X)[:, 1]

    def save(self, path: str | Path = config.MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path = config.MODEL_PATH) -> "FakeNewsModel":
        return joblib.load(path)


def train_model(texts: list[str], labels) -> FakeNewsModel:
    """Fit the vectorizer, calibrated classifier, and explainer classifier."""
    y = np.asarray(labels)
    if len(np.unique(y)) < 2:
        raise ValueError("Training data must contain both real and fake examples.")

    vectorizer = TfidfVectorizer(**config.TFIDF_PARAMS)
    X = vectorizer.fit_transform(texts)

    folds = min(config.CALIBRATION_FOLDS, int(np.bincount(y).min()))
    if folds < 2:
        raise ValueError("Each class needs at least 2 examples to calibrate the model.")

    calibrated = CalibratedClassifierCV(
        LogisticRegression(**config.LOGREG_PARAMS),
        method="sigmoid",
        cv=folds,
    )
    calibrated.fit(X, y)

    explainer = LogisticRegression(**config.LOGREG_PARAMS)
    explainer.fit(X, y)

    return FakeNewsModel(vectorizer=vectorizer, calibrated=calibrated, explainer=explainer)
