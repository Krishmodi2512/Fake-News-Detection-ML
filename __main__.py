"""Command-line usage: python -m fakenews "headline text" """
import argparse
import json

from fakenews.predict import load_model, predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a news headline or article.")
    parser.add_argument("text", nargs="?", help="Text to analyze. Reads stdin if omitted.")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    args = parser.parse_args()

    text = args.text if args.text is not None else input("Enter text: ")
    result = predict(load_model(), text)

    if args.json:
        print(json.dumps(result.__dict__, indent=2))
        return

    print(f"Prediction:  {result.label}")
    print(f"Confidence:  {result.confidence:.1%}")
    if result.low_confidence:
        print("Note:        Low confidence. Treat this result as inconclusive.")
    for w in result.warnings:
        print(f"Warning:     {w}")
    print("Style flags:")
    for f in result.style["flags"]:
        print(f"  - {f}")
    print(result.disclaimer)


if __name__ == "__main__":
    main()
