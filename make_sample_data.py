"""Generate a small SYNTHETIC demo dataset for testing the pipeline.

This data is templated and is NOT a benchmark. Models trained on it will
look accurate only because the templates are easy to separate. For real
evaluation, use a dataset such as ISOT, LIAR, or FakeNewsNet (see README).
"""
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fakenews import config  # noqa: E402

SLOTS = {
    "actor": ["The governor", "A senior official", "The health ministry", "A tech billionaire",
              "The city council", "Local regulators", "The central bank", "A state agency"],
    "object": ["the new tax plan", "the bridge contract", "the vaccine data", "the election results",
               "the housing budget", "the school funding formula", "the transit proposal", "the drought report"],
    "thing": ["This supplement", "This household gadget", "This tea", "This app", "This kitchen trick"],
    "claim": ["melts belly fat overnight", "reverses aging in one week", "cures every illness"],
    "agency": ["The labor department", "The transportation agency", "The statistics office", "The water board"],
    "journal": ["the Journal of Applied Economics", "a peer-reviewed public health journal",
                "the International Review of Transport Studies"],
    "finding": ["unemployment fell slightly last quarter", "bus ridership rose 4 percent",
                "rainfall totals were below the seasonal average", "hospital wait times declined"],
}

FAKE_TEMPLATES = [
    "BREAKING: {actor} is secretly hiding {object}, insiders say!!!",
    "You won't believe what {actor} just did with {object}. Share before it gets deleted!",
    "SHOCKING: {thing} {claim}, and the media refuses to report it. Wake up people!",
    "Doctors HATE this one trick that {claim}. Big pharma doesn't want you to know!",
    "EXPOSED: {actor} caught in a massive cover-up over {object}. The truth is explosive!",
    "Leaked documents prove {actor} plans to destroy {object}. Share this now!!!",
    "Unbelievable! {actor} and {object} are a hoax, and nobody is talking about it!",
    "Miracle discovery: {thing} {claim}. The government banned it. Share before it's gone!",
]

REAL_TEMPLATES = [
    "{agency} said on Tuesday that {finding}, according to its quarterly report.",
    "A new study in {journal} found that {finding}, the researchers reported.",
    "{actor} released a statement on {object} after a months-long public review, officials said.",
    "The council approved funding for {object} on Thursday, local officials confirmed.",
    "{agency} published data showing that {finding}, a spokesperson told reporters.",
    "{actor} said the decision on {object} will take effect next spring, according to a filing.",
    "Officials said {object} is expected to be reviewed again next year, the statement added.",
]


def generate(n_per_class: int, seed: int = 7) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    rows: list[tuple[str, int]] = []
    for templates, label in ((FAKE_TEMPLATES, 1), (REAL_TEMPLATES, 0)):
        count = 0
        attempts = 0
        while count < n_per_class and attempts < n_per_class * 50:
            attempts += 1
            slots = {k: rng.choice(v) for k, v in SLOTS.items()}
            text = rng.choice(templates).format(**slots)
            if text not in seen:
                seen.add(text)
                rows.append((text, label))
                count += 1
    rng.shuffle(rows)
    return rows


def main() -> None:
    rows = generate(n_per_class=300)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SAMPLE_DATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic rows to {config.SAMPLE_DATA_PATH}")


if __name__ == "__main__":
    main()
