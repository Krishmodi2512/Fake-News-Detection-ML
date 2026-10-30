"""Train, evaluate, and save the model.

Usage:
    python scripts/train.py                       # uses data/sample_news.csv
    python scripts/train.py --data path/to.csv
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.model_selection import train_test_split  # noqa: E402

from fakenews import config  # noqa: E402
from fakenews.data import load_dataset  # noqa: E402
from fakenews.evaluate import evaluate_model  # noqa: E402
from fakenews.model import train_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(config.SAMPLE_DATA_PATH))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model-out", default=str(config.MODEL_PATH))
    parser.add_argument("--metrics-out", default=str(config.METRICS_PATH))
    args = parser.parse_args()

    df = load_dataset(args.data)
    print(f"Loaded {len(df)} examples | fake share: {df['label'].mean():.1%}")

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, stratify=df["label"], random_state=config.RANDOM_STATE
    )

    model = train_model(train_df["text"].tolist(), train_df["label"].tolist())
    metrics = evaluate_model(model, test_df["text"].tolist(), test_df["label"].tolist())
    metrics["dataset"] = str(args.data)
    metrics["train_size"] = int(len(train_df))

    model.save(args.model_out)
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))

    print(json.dumps({k: v for k, v in metrics.items() if k != "calibration_curve"}, indent=2))
    print(f"Model saved to {args.model_out}")
    print(f"Metrics saved to {args.metrics_out}")


if __name__ == "__main__":
    main()
