"""End-to-end prediction: validate input, score it, and attach analysis."""
from dataclasses import dataclass, field

from fakenews import config
from fakenews.explain import top_contributions
from fakenews.model import FakeNewsModel
from fakenews.preprocessing import validate_input
from fakenews.signals import analyze_style

LABEL_REAL_TEXT = "Likely Real"
LABEL_FAKE_TEXT = "Potentially Fake"

DISCLAIMER = (
    "This is a statistical estimate based on writing patterns in the training data. "
    "It does not verify facts. Check the claim against reliable sources."
)


@dataclass
class Prediction:
    label: str
    confidence: float          # probability of the predicted label, 0.5 to 1.0
    p_fake: float              # calibrated probability the text is fake
    low_confidence: bool
    warnings: list[str] = field(default_factory=list)
    style: dict = field(default_factory=dict)
    terms: dict = field(default_factory=dict)
    disclaimer: str = DISCLAIMER


def predict(model: FakeNewsModel, raw_text: str) -> Prediction:
    cleaned, warnings = validate_input(raw_text)
    p_fake = float(model.predict_proba_fake([cleaned])[0])

    if p_fake >= config.DECISION_THRESHOLD:
        label, confidence = LABEL_FAKE_TEXT, p_fake
    else:
        label, confidence = LABEL_REAL_TEXT, 1.0 - p_fake

    return Prediction(
        label=label,
        confidence=round(confidence, 4),
        p_fake=round(p_fake, 4),
        low_confidence=confidence < config.LOW_CONFIDENCE_THRESHOLD,
        warnings=warnings,
        style=analyze_style(cleaned),
        terms=top_contributions(model, cleaned),
    )


def load_model(path=config.MODEL_PATH) -> FakeNewsModel:
    return FakeNewsModel.load(path)
