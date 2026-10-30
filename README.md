# Fake News Detector

A text classifier that estimates whether a news headline or article is **Likely Real** or **Potentially Fake**. It reports a calibrated confidence score and the writing signals and terms behind the prediction.

> **This tool does not verify facts.** It learns patterns in writing style and word usage from its training data. A well-written false story can pass, and an unusual true story can be flagged. Use the output as one input to your judgment, not as a verdict.

## Features

- TF-IDF (words and bigrams) with logistic regression
- Sigmoid (Platt) calibration, so the displayed confidence is closer to the observed accuracy
- Per-term contributions showing which words pushed the score toward fake or real
- Style analysis: ALL-CAPS ratio, exclamation marks, sensational phrasing, source attribution
- Input validation: empty input is rejected, short input and truncation are flagged
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, Brier score, confusion matrix, calibration curve
- Streamlit web UI, command-line tool, and tests

## Project layout

```
fake-news-detector/
├── app/streamlit_app.py      # web interface
├── src/fakenews/
│   ├── config.py             # paths, thresholds, hyperparameters
│   ├── data.py               # CSV loading and label normalization
│   ├── preprocessing.py      # text cleaning and input validation
│   ├── signals.py            # style analysis (shown to user, not a model input)
│   ├── model.py              # TF-IDF + calibrated logistic regression
│   ├── explain.py            # per-term contributions
│   ├── evaluate.py           # metrics and calibration curve
│   ├── predict.py            # end-to-end prediction
│   └── __main__.py           # CLI
├── scripts/
│   ├── make_sample_data.py   # generates synthetic demo data
│   └── train.py              # train, evaluate, save
├── tests/                    # pytest suite
├── data/sample_news.csv      # SYNTHETIC demo data (not a benchmark)
├── models/                   # trained model artifact (created by train.py)
└── reports/metrics.json      # evaluation output (created by train.py)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # makes the `fakenews` package importable everywhere
```

## Quick start

```bash
python scripts/train.py                              # train on the bundled sample data
python -m streamlit run app/streamlit_app.py         # launch the web UI
python -m fakenews "Scientists confirm miracle cure, share before it gets deleted!!!"
pytest                                               # run tests
```

## Using your own dataset

Prepare a CSV with these columns:

| column  | required | description                                              |
|---------|----------|----------------------------------------------------------|
| `text`  | yes      | article body, or the headline if that is all you have   |
| `title` | no       | prepended to `text` when present                         |
| `label` | yes      | `1` or `fake` = fake; `0` or `real` = real               |

Then train on it:

```bash
python scripts/train.py --data path/to/your_data.csv
```

Suggested public datasets:

- **ISOT Fake News Dataset** (Kaggle): article-level, separate `True.csv` and `Fake.csv` files. Combine them and add a `label` column.
- **LIAR** (PolitiFact): claim-level, six truthfulness labels. Map them to binary (e.g., pants-on-fire/false/barely-true = fake).
- **FakeNewsNet** (GossipCop and PolitiFact): article-level with social context.

## About the sample data

`data/sample_news.csv` is generated from templates by `scripts/make_sample_data.py`. It exists only so the pipeline runs end to end. Its near-perfect metrics reflect how easy the templates are to separate, not real-world performance. Always evaluate on a real dataset before relying on the model.

## Evaluation and limitations

- **Dataset shift.** Models trained on one source often drop in accuracy on other sources, topics, or time periods. Test on a source held out from training.
- **Calibration.** Probabilities are calibrated on training folds. Check `reports/metrics.json` (`brier_score` and `calibration_curve`) on your real data. Recalibrate if the curve departs from the diagonal.
- **Style is not truth.** Real news written in a sensational style gets flagged, and careful fake news gets through.
- **Language.** The model handles English text only, and the stop-word list is not applied, so function words are included as features.

## Extending the project

- Replace TF-IDF with a fine-tuned transformer (e.g., DistilBERT) for higher accuracy, at the cost of more compute.
- Add a source-reputation or claim-matching step that checks against fact-checking databases. This is the only way to move beyond style-based signals.
- Add a REST API (FastAPI) around `predict()` for integration with other services.
