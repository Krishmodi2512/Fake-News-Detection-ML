"""Evaluation metrics, including calibration quality."""
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fakenews import config
from fakenews.model import FakeNewsModel


def evaluate_model(model: FakeNewsModel, texts: list[str], labels) -> dict:
    y = np.asarray(labels)
    p_fake = model.predict_proba_fake(texts)
    y_pred = (p_fake >= config.DECISION_THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    try:
        auc = float(roc_auc_score(y, p_fake))
    except ValueError:  # only one class present in the evaluation set
        auc = None

    frac_pos, mean_pred = calibration_curve(y, p_fake, n_bins=10, strategy="uniform")

    return {
        "n_examples": int(len(y)),
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "precision_fake": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall_fake": round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1_fake": round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "brier_score": round(float(brier_score_loss(y, p_fake)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "calibration_curve": [
            {"mean_predicted": round(float(a), 4), "observed_fraction": round(float(b), 4)}
            for a, b in zip(mean_pred, frac_pos)
        ],
    }
