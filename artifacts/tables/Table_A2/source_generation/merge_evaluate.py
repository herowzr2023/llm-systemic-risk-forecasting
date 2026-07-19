"""Merge one or more tidy classifier-evaluation CSV files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = ["dataset", "model_name", "accuracy", "precision", "recall", "f1"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True,
                        help="Tidy evaluation CSV; repeat this option to merge several files.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frames = []
    for path in args.input:
        frame = pd.read_csv(path)
        missing = set(COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(frame[COLUMNS])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False, encoding="utf-8-sig")
    print(output)


if __name__ == "__main__":
    main()
