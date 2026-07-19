from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterable

import pandas as pd

from common import read_csv_auto, resolve_path, write_csv


LABEL_TO_TEXT = {-1: "负面", 0: "中性", 1: "正面"}
TEXT_TO_LABEL = {
    "-1": -1,
    "负面": -1,
    "negative": -1,
    "0": 0,
    "中性": 0,
    "neutral": 0,
    "1": 1,
    "+1": 1,
    "正面": 1,
    "positive": 1,
}

DEFAULT_INSTRUCTION = (
    "下面语句的情绪是什么？必须从括号内选项选择回答 {正面/中性/负面}，"
    "只回答正面、中性或负面。"
)

SENTIMENT_RAW_COLUMNS = [
    "Negative_Count",
    "Neutral_Count",
    "Positive_Count",
    "Negative_read_num",
    "Neutral_read_num",
    "Positive_read_num",
    "Negative_comments_num",
    "Neutral_comments_num",
    "Positive_comments_num",
    "read_num",
    "comments_num",
]


@dataclass(frozen=True)
class LabelSplitConfig:
    base_csv: Path
    output_dir: Path
    text_col: str = "text"
    label_col: str = "sen"
    validation_size: float = 0.1
    random_state: int = 42
    stratify: bool = True
    instruction: str = DEFAULT_INSTRUCTION


@dataclass(frozen=True)
class PredictionAggregationConfig:
    prediction_glob: str
    merged_predictions_csv: Path
    fixed_predictions_csv: Path
    final_stats_csv: Path
    date_col: str = "pub_date"
    bank_col: str = "bank_code"
    prediction_col: str = "predict"
    read_col: str = "read_num"
    comment_col: str = "comments_num"
    date_corrections: dict[int, str] | None = None


def normalize_label(value: object) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("Missing sentiment label")
    text = str(value).strip()
    key = text.lower()
    if key in TEXT_TO_LABEL:
        return TEXT_TO_LABEL[key]
    if text in TEXT_TO_LABEL:
        return TEXT_TO_LABEL[text]
    numeric = int(float(text))
    if numeric not in LABEL_TO_TEXT:
        raise ValueError(f"Unsupported sentiment label: {value!r}")
    return numeric


def prediction_from_response(value: object) -> int:
    text = "" if value is None or pd.isna(value) else str(value)
    if "负面" in text:
        return -1
    if "正面" in text:
        return 1
    if "中性" in text:
        return 0
    return normalize_label(text)


def _split_labeled_frame(df: pd.DataFrame, cfg: LabelSplitConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.text_col not in df.columns or cfg.label_col not in df.columns:
        raise ValueError(f"Required columns not found: {cfg.text_col!r}, {cfg.label_col!r}")
    df = df.copy()
    df[cfg.label_col] = df[cfg.label_col].apply(normalize_label)

    if not cfg.stratify:
        val = df.sample(frac=cfg.validation_size, random_state=cfg.random_state)
        train = df.drop(val.index)
        return train.reset_index(drop=True), val.reset_index(drop=True)

    val_parts = []
    for _, group in df.groupby(cfg.label_col, sort=False):
        n_val = max(1, int(round(len(group) * cfg.validation_size)))
        val_parts.append(group.sample(n=n_val, random_state=cfg.random_state))
    val = pd.concat(val_parts).sort_index()
    train = df.drop(val.index)
    return train.reset_index(drop=True), val.reset_index(drop=True)


def _to_instruction_json(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    instruction: str,
) -> list[dict[str, str]]:
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "instruction": instruction,
                "input": "" if pd.isna(row[text_col]) else str(row[text_col]),
                "output": LABEL_TO_TEXT[normalize_label(row[label_col])],
            }
        )
    return records


def split_labeled_data(cfg: LabelSplitConfig) -> dict[str, Path]:
    df = read_csv_auto(cfg.base_csv)
    train, val = _split_labeled_frame(df, cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = write_csv(train, cfg.output_dir / "train.csv")
    val_csv = write_csv(val, cfg.output_dir / "val.csv")

    train_json_dir = cfg.output_dir / "train_json"
    validation_json_dir = cfg.output_dir / "validation_json"
    train_json_dir.mkdir(parents=True, exist_ok=True)
    validation_json_dir.mkdir(parents=True, exist_ok=True)
    train_json = train_json_dir / "train_output.json"
    val_json = validation_json_dir / "val_output.json"
    train_json.write_text(
        json.dumps(_to_instruction_json(train, cfg.text_col, cfg.label_col, cfg.instruction), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    val_json.write_text(
        json.dumps(_to_instruction_json(val, cfg.text_col, cfg.label_col, cfg.instruction), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"train_csv": train_csv, "val_csv": val_csv, "train_json": train_json, "val_json": val_json}


def prediction_paths(pattern: str) -> list[Path]:
    return [Path(p) for p in sorted(glob(str(resolve_path(pattern))))]


def audit_prediction_inputs(pattern: str) -> pd.DataFrame:
    rows = []
    for path in prediction_paths(pattern):
        rows.append({"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0})
    return pd.DataFrame(rows)


def merge_prediction_files(pattern: str, output_csv: Path) -> Path:
    paths = prediction_paths(pattern)
    if not paths:
        raise FileNotFoundError(f"No prediction CSV matched pattern: {pattern}")
    frames = [read_csv_auto(path, low_memory=False) for path in paths]
    merged = pd.concat(frames, ignore_index=True)
    return write_csv(merged, output_csv)


def parse_count(value: object) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(float(value))
    text = str(value).strip().replace(",", "")
    if text == "" or text.lower() in {"nan", "none", "--"}:
        return 0
    multiplier = 1.0
    if "亿" in text:
        multiplier = 100000000.0
    elif "万" in text:
        multiplier = 10000.0
    text = text.replace("亿", "").replace("万", "").replace("+", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return int(float(match.group(0)) * multiplier) if match else 0


def build_daily_sentiment_stats(cfg: PredictionAggregationConfig) -> dict[str, Path]:
    if not cfg.merged_predictions_csv.exists():
        merge_prediction_files(cfg.prediction_glob, cfg.merged_predictions_csv)

    df = read_csv_auto(cfg.merged_predictions_csv, low_memory=False)
    required = {cfg.date_col, cfg.bank_col, cfg.prediction_col, cfg.read_col, cfg.comment_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    if cfg.date_corrections:
        for row_index, corrected_date in cfg.date_corrections.items():
            if row_index in df.index:
                df.loc[row_index, cfg.date_col] = corrected_date

    df[cfg.read_col] = df[cfg.read_col].apply(parse_count).astype(int)
    df[cfg.comment_col] = df[cfg.comment_col].apply(parse_count).astype(int)
    df[cfg.prediction_col] = df[cfg.prediction_col].apply(normalize_label)
    df["pub_day"] = pd.to_datetime(df[cfg.date_col], errors="coerce").dt.date
    df = df.dropna(subset=[cfg.bank_col, "pub_day", cfg.prediction_col])

    fixed_path = write_csv(df, cfg.fixed_predictions_csv)
    group_cols = [cfg.bank_col, "pub_day"]

    counts = (
        df.groupby(group_cols + [cfg.prediction_col])[cfg.prediction_col]
        .count()
        .unstack(fill_value=0)
        .reindex(columns=[-1, 0, 1], fill_value=0)
        .rename(columns={-1: "Negative_Count", 0: "Neutral_Count", 1: "Positive_Count"})
    )

    sums = (
        df.groupby(group_cols + [cfg.prediction_col])[[cfg.read_col, cfg.comment_col]]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=pd.MultiIndex.from_product([[cfg.read_col, cfg.comment_col], [-1, 0, 1]]), fill_value=0)
    )
    rename_label = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    sums.columns = [f"{rename_label[int(label)]}_{metric}" for metric, label in sums.columns]

    totals = df.groupby(group_cols)[[cfg.read_col, cfg.comment_col]].sum()
    final_stats = counts.merge(sums, left_index=True, right_index=True, how="outer")
    final_stats = final_stats.merge(totals, left_index=True, right_index=True, how="outer")
    final_stats = final_stats.reindex(columns=SENTIMENT_RAW_COLUMNS, fill_value=0)
    final_stats = final_stats.reset_index()
    final_stats = final_stats[[cfg.bank_col, "pub_day", *SENTIMENT_RAW_COLUMNS]]
    final_stats_path = write_csv(final_stats, cfg.final_stats_csv)
    return {"fixed_predictions_csv": fixed_path, "final_stats_csv": final_stats_path}
