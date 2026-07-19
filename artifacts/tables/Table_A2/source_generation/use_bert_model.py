"""Apply a fine-tuned BERT sentiment classifier to a CSV in-process."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    from transformers import pipeline
    frame = pd.read_csv(args.input)
    classifier = pipeline("text-classification", model=args.model, tokenizer=args.model,
                          device=0, batch_size=args.batch_size, truncation=True, max_length=256)
    frame["prediction"] = [result["label"] for result in classifier(frame[args.text_column].fillna("").astype(str).tolist())]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    print(output)


if __name__ == "__main__":
    main()

