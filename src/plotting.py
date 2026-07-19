from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import read_csv_auto


AVAILABLE_FONTS = {font.name for font in font_manager.fontManager.ttflist}
PREFERRED_FONTS = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["font.sans-serif"] = [font for font in PREFERRED_FONTS if font in AVAILABLE_FONTS] or ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "x", "P", "*", "v", "<", ">"]

BANK_NAME_MAP = {
    "工商银行": "ICBC",
    "建设银行": "CCB",
    "中国银行": "BOC",
    "农业银行": "ABC",
    "交通银行": "BCM",
    "招商银行": "CMB",
    "中信银行": "CNCB",
    "兴业银行": "CIB",
    "民生银行": "CMBC",
    "浦发银行": "SPDB",
    "平安银行": "PAB",
    "宁波银行": "NBCB",
    "南京银行": "BON",
    "北京银行": "BOB",
    "华夏银行": "HB",
}


def _prepare_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out.sort_values(date_col)


def save_model_sequence_plot(
    prediction_csv: Path,
    output_png: Path,
    actual_col: str = "TRUE",
    date_col: str = "date",
    baseline_col: str = "baseline_pred",
    model_columns: list[str] | None = None,
) -> Path:
    df = _prepare_dates(read_csv_auto(prediction_csv), date_col)
    pred_cols = model_columns or [baseline_col] + [c for c in df.columns if c.endswith("_pred") and c != baseline_col]
    missing = [column for column in pred_cols if column not in df.columns]
    if missing:
        raise KeyError(f"Prediction columns not found in {prediction_csv}: {missing}")
    rows = ceil(len(pred_cols) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), sharex=False)
    axes = np.asarray(axes).reshape(-1)

    for idx, col in enumerate(pred_cols):
        ax = axes[idx]
        model_name = "Informer" if col == baseline_col else col.replace("_pred", "")
        model_name = {
            "GradientBoostingRegressor": "Gradient Boosting",
            "AR1": "AR(1)",
            "ETS_ANN": "Exponential Smoothing",
        }.get(model_name, model_name)
        ax.plot(df[date_col], df[actual_col], color="black", linewidth=1.4, linestyle="-", label="True")
        ax.plot(
            df[date_col],
            df[col],
            color="black",
            linewidth=1.2,
            linestyle=LINE_STYLES[idx % len(LINE_STYLES)],
            marker=MARKERS[idx % len(MARKERS)],
            markersize=2.4,
            markevery=max(1, len(df) // 25),
            label="Predicted",
        )
        ax.set_title(model_name)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel("Date")
        ax.set_ylabel("SR proxy")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

    for idx in range(len(pred_cols), len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_png


def save_time_series_panel(
    input_csv: Path,
    output_png: Path,
    columns: list[str],
    labels: list[str] | None = None,
    date_col: str = "date",
    resample_rule: str = "ME",
) -> Path:
    df = _prepare_dates(read_csv_auto(input_csv), date_col).set_index(date_col)
    selected = df[columns].apply(pd.to_numeric, errors="coerce")
    selected = (selected - selected.mean()) / selected.std()
    if resample_rule:
        selected = selected.resample(resample_rule).mean()
    labels = labels or columns

    rows = ceil(len(columns) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 3.2 * rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for idx, (col, label) in enumerate(zip(columns, labels)):
        ax = axes[idx]
        ax.plot(
            selected.index,
            selected[col],
            color="black",
            linestyle=LINE_STYLES[idx % len(LINE_STYLES)],
            marker=MARKERS[idx % len(MARKERS)],
            markersize=2.2,
            markevery=max(1, len(selected) // 24),
            linewidth=1.2,
        )
        ax.set_title(label)
        ax.set_xlabel("Date")
        ax.set_ylabel("Standardized value")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for idx in range(len(columns), len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_png


def save_bank_trend_grid(best_results_dir: Path, output_png: Path, horizon: int, sentiment: str = "with_sen") -> Path:
    files = sorted(best_results_dir.glob(f"*_best_results_{sentiment}_for_{horizon}.csv"))
    files = [path for path in files if not path.name.startswith("ALL_")]
    if not files:
        raise FileNotFoundError(f"No bank best-result files found in {best_results_dir} for horizon={horizon}, sentiment={sentiment}")

    rows = ceil(len(files) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 3.2 * rows), sharex=False)
    axes = np.asarray(axes).reshape(-1)
    for idx, path in enumerate(files):
        bank = path.name.split("_best_results_")[0]
        bank = BANK_NAME_MAP.get(bank, bank)
        df = _prepare_dates(read_csv_auto(path), "date")
        ax = axes[idx]
        ax.plot(df["date"], df["true"], color="black", linewidth=1.2, linestyle="-", label="True")
        ax.plot(
            df["date"],
            df["pred"],
            color="black",
            linewidth=1.1,
            linestyle=LINE_STYLES[(idx + 1) % len(LINE_STYLES)],
            marker=MARKERS[idx % len(MARKERS)],
            markersize=2.2,
            markevery=max(1, len(df) // 20),
            label="Predicted",
        )
        ax.set_title(bank)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel("Date")
        ax.set_ylabel("SR proxy")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    for idx in range(len(files), len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_png
