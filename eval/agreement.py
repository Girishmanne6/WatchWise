"""Compare hand labels in eval/manual_check.csv with the judge, once the empty columns are filled in."""

import sys
from pathlib import Path

import pandas as pd

PATH = Path(__file__).resolve().parent / "manual_check.csv"
PAIRS = [
    ("judge_hallucination", "human_hallucination"),
    ("judge_spoiler", "human_spoiler"),
    ("judge_verdict", "human_verdict"),
]


def main() -> None:
    if not PATH.exists():
        print(f"missing {PATH}")
        sys.exit(1)
    frame = pd.read_csv(PATH)
    for judge_col, human_col in PAIRS:
        if human_col not in frame.columns:
            print(f"{human_col} is not in the file")
            continue
        labeled = frame[frame[human_col].notna() & (frame[human_col].astype(str).str.strip() != "")]
        if labeled.empty:
            print(f"{human_col}: no hand labels yet")
            continue
        left = labeled[judge_col].astype(str).str.strip().str.lower()
        right = labeled[human_col].astype(str).str.strip().str.lower()
        agree = (left == right).mean()
        print(f"{human_col}: {agree:.1%} agreement on {len(labeled)} rows")


if __name__ == "__main__":
    main()
