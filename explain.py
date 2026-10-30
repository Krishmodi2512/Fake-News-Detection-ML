"""Per-term contributions to the prediction (linear model: contribution = tf-idf * coefficient)."""
import numpy as np

from fakenews.model import FakeNewsModel


def top_contributions(model: FakeNewsModel, text: str, k: int = 8) -> dict:
    """Terms in this text that pushed the score toward fake or toward real."""
    X = model.vectorizer.transform([text]).tocsr()
    if X.nnz == 0:
        return {"toward_fake": [], "toward_real": []}

    coefs = model.explainer.coef_[0]
    names = model.vectorizer.get_feature_names_out()
    idx = X.indices
    contrib = X.data * coefs[idx]

    order = np.argsort(contrib)
    toward_fake = [
        (str(names[idx[i]]), round(float(contrib[i]), 4))
        for i in order[::-1] if contrib[i] > 0
    ][:k]
    toward_real = [
        (str(names[idx[i]]), round(float(contrib[i]), 4))
        for i in order if contrib[i] < 0
    ][:k]
    return {"toward_fake": toward_fake, "toward_real": toward_real}
