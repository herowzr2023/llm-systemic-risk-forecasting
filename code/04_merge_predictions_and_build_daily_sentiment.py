from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import load_json, resolve_path, write_csv
from sentiment_data import PredictionAggregationConfig, audit_prediction_inputs, build_daily_sentiment_stats, merge_prediction_files


def build_config(raw: dict) -> PredictionAggregationConfig:
    return PredictionAggregationConfig(
        prediction_glob=raw["prediction_glob"],
        merged_predictions_csv=resolve_path(raw["merged_predictions_csv"]),
        fixed_predictions_csv=resolve_path(raw["fixed_predictions_csv"]),
        final_stats_csv=resolve_path(raw["final_stats_csv"]),
        date_col=raw.get("date_col", "pub_date"),
        bank_col=raw.get("bank_col", "bank_code"),
        prediction_col=raw.get("prediction_col", "predict"),
        read_col=raw.get("read_col", "read_num"),
        comment_col=raw.get("comment_col", "comments_num"),
        date_corrections={int(k): v for k, v in raw.get("date_corrections", {}).items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LLM prediction CSVs and build institution-day sentiment indicators.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "04_daily_sentiment.json"))
    parser.add_argument("--step", choices=["audit", "merge", "daily-stats", "all"], default="all")
    args = parser.parse_args()

    raw = load_json(args.config)
    cfg = build_config(raw)
    if args.step in {"audit", "all"}:
        audit = audit_prediction_inputs(cfg.prediction_glob)
        audit_path = resolve_path(raw.get("audit_csv", "outputs/prediction_input_audit.csv"))
        write_csv(audit, audit_path)
        print(f"audit_csv: {audit_path}")
    if args.step in {"merge", "all"}:
        merged = merge_prediction_files(cfg.prediction_glob, cfg.merged_predictions_csv)
        print(f"merged_predictions_csv: {merged}")
    if args.step in {"daily-stats", "all"}:
        outputs = build_daily_sentiment_stats(cfg)
        for name, path in outputs.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
