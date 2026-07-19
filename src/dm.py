from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from common import read_csv_auto, write_csv


MODEL_LABELS = {"Informer": "Informer", "Persistence": "Persistence", "AR1": "AR(1)",
                "ETS": "ETS", "GBR": "GBR", "PatchTST": "PatchTST"}


def _model_label(column: str) -> str:
    key = column.removesuffix("_pred")
    return MODEL_LABELS.get(key, key)


def _validate(frame: pd.DataFrame, actual_col: str, baseline_col: str, date_col: str) -> tuple[pd.DataFrame, list[str]]:
    required = {date_col, actual_col, baseline_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    data = frame.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="raise")
    if data.duplicated(date_col).any():
        raise ValueError("The reported comparison requires one observation per target date")
    predictions = [column for column in data.columns if column.endswith("_pred")]
    if baseline_col not in predictions or len(predictions) < 2:
        raise ValueError("A baseline and at least one comparator are required")
    for column in [actual_col, *predictions]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column]).all():
            raise ValueError(f"Non-finite values in {column}")
    return data.sort_values(date_col).reset_index(drop=True), predictions


def _significance(p_value: float) -> str:
    return "***" if p_value < .01 else "**" if p_value < .05 else "*" if p_value < .10 else ""


def dm_statistic(actual: np.ndarray, baseline: np.ndarray, comparator: np.ndarray,
                 absolute: bool) -> tuple[float, float]:
    """Paired DM statistic for the paper's target-date loss sequence.

    The loss differential is Informer loss minus comparator loss. A negative
    statistic therefore favors Informer. The archived experiment treats the
    date-level series as the evaluation sequence and applies the
    Harvey-Leybourne-Newbold finite-sample correction with effective horizon
    one. This function reproduces that reported design exactly.
    """
    baseline_error = actual - baseline
    comparator_error = actual - comparator
    differential = (np.abs(baseline_error) - np.abs(comparator_error) if absolute
                    else baseline_error**2 - comparator_error**2)
    nobs = len(differential)
    mean = float(differential.mean())
    centered = differential - mean
    gamma0 = float(np.sum(centered**2) / nobs)
    variance_mean = gamma0 / nobs
    if variance_mean <= 0 or not math.isfinite(variance_mean):
        raise ValueError("The loss-differential variance is not positive")
    statistic = mean / math.sqrt(variance_mean) * math.sqrt((nobs - 1) / nobs)
    p_value = float(2 * student_t.cdf(-abs(statistic), df=nobs - 1))
    return statistic, p_value


def calculate_metrics(frame: pd.DataFrame, actual_col: str, predictions: list[str]) -> pd.DataFrame:
    actual = frame[actual_col].to_numpy(float)
    denominator = float(np.sum((actual - actual.mean())**2))
    rows = []
    for column in predictions:
        error = actual - frame[column].to_numpy(float)
        mse = float(np.mean(error**2))
        rows.append({"model": _model_label(column), "n": len(actual), "MSE": mse,
                     "RMSE": math.sqrt(mse), "MAE": float(np.mean(np.abs(error))),
                     "R2": float(1 - np.sum(error**2) / denominator)})
    return pd.DataFrame(rows)


def dm_table(frame: pd.DataFrame, actual_col: str, baseline_col: str,
             predictions: list[str]) -> pd.DataFrame:
    actual = frame[actual_col].to_numpy(float)
    baseline = frame[baseline_col].to_numpy(float)
    records = []
    for criterion, absolute in (("MAD", True), ("MSE", False)):
        row = {"criterion": criterion}
        for column in predictions:
            if column == baseline_col:
                continue
            statistic, p_value = dm_statistic(actual, baseline, frame[column].to_numpy(float), absolute)
            row[column.removesuffix("_pred")] = f"{statistic:.3f}{_significance(p_value)}"
        records.append(row)
    return pd.DataFrame(records)


def analyze_prediction_file(input_csv: Path, metrics_output_csv: Path, dm_output_csv: Path,
                            actual_col: str = "TRUE", baseline_col: str = "Informer_pred",
                            date_col: str = "date") -> dict[str, Path]:
    frame, predictions = _validate(read_csv_auto(input_csv), actual_col, baseline_col, date_col)
    metrics_path = write_csv(calculate_metrics(frame, actual_col, predictions), metrics_output_csv)
    dm_path = write_csv(dm_table(frame, actual_col, baseline_col, predictions), dm_output_csv)
    return {"metrics_output_csv": metrics_path, "dm_output_csv": dm_path}
