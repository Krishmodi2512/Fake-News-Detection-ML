"""Central configuration for paths, model settings, and thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "fake_news_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
SAMPLE_DATA_PATH = DATA_DIR / "sample_news.csv"

RANDOM_STATE = 42

# Label convention: 1 = fake (positive class), 0 = real.
LABEL_REAL = 0
LABEL_FAKE = 1
DECISION_THRESHOLD = 0.5

# Shown to users when the model is unsure. Probability of the predicted label below this is flagged.
LOW_CONFIDENCE_THRESHOLD = 0.65

MIN_WORDS = 5
MAX_CHARS = 10_000

TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    min_df=2,
    max_features=50_000,
    sublinear_tf=True,
    lowercase=True,
)
LOGREG_PARAMS = dict(C=1.0, max_iter=2000, class_weight="balanced")
CALIBRATION_FOLDS = 5
