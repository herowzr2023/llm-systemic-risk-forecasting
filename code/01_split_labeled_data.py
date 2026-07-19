from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import load_json, resolve_path
from sentiment_data import DEFAULT_INSTRUCTION, LabelSplitConfig, split_labeled_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Split GPT-4o pseudo-labelled sentiment data into train/validation files.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "01_label_split.json"))
    args = parser.parse_args()

    raw = load_json(args.config)
    cfg = LabelSplitConfig(
        base_csv=resolve_path(raw["base_csv"]),
        output_dir=resolve_path(raw["output_dir"]),
        text_col=raw.get("text_col", "text"),
        label_col=raw.get("label_col", "sen"),
        validation_size=float(raw.get("validation_size", 0.1)),
        random_state=int(raw.get("random_state", 42)),
        stratify=bool(raw.get("stratify", True)),
        instruction=raw.get("instruction", DEFAULT_INSTRUCTION),
    )
    outputs = split_labeled_data(cfg)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
