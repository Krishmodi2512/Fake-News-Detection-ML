import numpy as np
import pytest

from fakenews import config
from fakenews.data import load_dataset
from fakenews.evaluate import evaluate_model
from fakenews.explain import top_contributions
from fakenews.model import FakeNewsModel, train_model
from fakenews.predict import LABEL_FAKE_TEXT, LABEL_REAL_TEXT, predict


@pytest.fixture(scope="module")
def trained():
    df = load_dataset(config.SAMPLE_DATA_PATH)
    return train_model(df["text"].tolist(), df["label"].tolist()), df


def test_probabilities_are_valid(trained):
    model, df = trained
    p = model.predict_proba_fake(df["text"].tolist()[:20])
    assert np.all((p >= 0) & (p <= 1))


def test_model_separates_classes_on_sample(trained):
    model, df = trained
    metrics = evaluate_model(model, df["text"].tolist(), df["label"].tolist())
    assert metrics["accuracy"] > 0.9  # synthetic data; see README


def test_train_requires_both_classes():
    with pytest.raises(ValueError):
        train_model(["a real story here", "another real one"], [0, 0])


def test_save_and_load_roundtrip(trained, tmp_path):
    model, df = trained
    path = tmp_path / "m.joblib"
    model.save(path)
    loaded = FakeNewsModel.load(path)
    sample = df["text"].tolist()[:5]
    np.testing.assert_allclose(model.predict_proba_fake(sample), loaded.predict_proba_fake(sample))


def test_predict_returns_valid_label(trained):
    model, _ = trained
    result = predict(model, "The city council approved the new transit budget on Thursday, officials said.")
    assert result.label in {LABEL_REAL_TEXT, LABEL_FAKE_TEXT}
    assert 0.5 <= result.confidence <= 1.0
    assert result.style and result.terms is not None


def test_contributions_structure(trained):
    model, _ = trained
    terms = top_contributions(model, "BREAKING shocking secret cover-up you won't believe this")
    assert set(terms) == {"toward_fake", "toward_real"}
    assert all(c > 0 for _, c in terms["toward_fake"])
    assert all(c < 0 for _, c in terms["toward_real"])
