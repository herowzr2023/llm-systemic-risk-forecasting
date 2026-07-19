from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr as scipy_pearsonr
except Exception:
    scipy_pearsonr = None
    from statistics import NormalDist

from common import read_csv_auto, write_csv


SENTIMENT_FEATURE_COLUMNS = [
    "Emotional_Skew",
    "Total_Count",
    "Weighted_Emotional_Skew_Read",
    "Weighted_Emotional_Skew_Comment",
]

DISPLAY_NAMES = {
    "Emotional_Skew": "Sentiment direction",
    "Total_Count": "Attention-based intensity",
    "Weighted_Emotional_Skew_Read": "Exposure-based intensity",
    "Weighted_Emotional_Skew_Comment": "Engagement-based intensity",
    "工商银行_CoVaR": "ICBC SR proxy",
    "ALL_CoVaR": "Banking sector SR proxy",
    "risk_spillover": "DY spillover To",
    "risk_receiving": "DY spillover From",
    "net_risk_value": "DY net spillover",
    "baseline": "Informer",
    "baseline_pred": "Informer",
    "GradientBoostingRegressor": "Gradient Boosting",
    "Persistence": "Persistence",
    "AR1": "AR(1)",
    "ETS_ANN": "ETS(A,N,N)",
    "Ridge ARX": "Ridge-ARX",
    "Random Forest Regressor": "Random Forest",
}


def _format_number(value: object, decimals: int = 2) -> str:
    return "" if pd.isna(value) else f"{float(value):.{decimals}f}"


def sentiment_descriptive_table(icbc_csv: Path, aggregate_csv: Path, output_csv: Path) -> Path:
    """Write Table 5 in the manuscript's two-row-header layout."""
    summaries: dict[str, pd.DataFrame] = {}
    for name, path in [("ICBC", icbc_csv), ("Banking sector", aggregate_csv)]:
        df = read_csv_auto(path)
        summary = (
            df[SENTIMENT_FEATURE_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .agg(["mean", "std", "min", "max"])
            .T
            .reindex(SENTIMENT_FEATURE_COLUMNS)
        )
        summaries[name] = summary

    columns = pd.MultiIndex.from_tuples(
        [("Index", "")]
        + [("ICBC", label) for label in ["Mean", "Standard", "Min", "Max"]]
        + [("Banking sector", label) for label in ["Mean", "Standard", "Min", "Max"]]
    )
    rows: list[list[str]] = []
    for feature in SENTIMENT_FEATURE_COLUMNS:
        rows.append(
            [DISPLAY_NAMES[feature]]
            + [_format_number(summaries["ICBC"].loc[feature, stat]) for stat in ["mean", "std", "min", "max"]]
            + [
                _format_number(summaries["Banking sector"].loc[feature, stat])
                for stat in ["mean", "std", "min", "max"]
            ]
        )
    return write_csv(pd.DataFrame(rows, columns=columns), output_csv)


def _format_corr(coef: float, p_value: float) -> str:
    if np.isnan(coef):
        return ""
    marker = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
    return f"{coef:.3f}{marker}"


def lagged_correlation_table(
    input_csv: Path,
    output_csv: Path,
    risk_cols: list[str],
    sentiment_cols: list[str] | None = None,
    lag: int = 1,
) -> Path:
    df = read_csv_auto(input_csv)
    sentiment_cols = sentiment_cols or SENTIMENT_FEATURE_COLUMNS
    rows = []
    for risk_col in risk_cols:
        for sentiment_col in sentiment_cols:
            pair = pd.DataFrame(
                {
                    "risk": pd.to_numeric(df[risk_col], errors="coerce"),
                    "sentiment_lag": pd.to_numeric(df[sentiment_col], errors="coerce").shift(lag),
                }
            ).dropna()
            if len(pair) < 3 or pair["risk"].nunique() <= 1 or pair["sentiment_lag"].nunique() <= 1:
                coef, p_value = np.nan, np.nan
            else:
                coef = float(pair["risk"].corr(pair["sentiment_lag"]))
                if scipy_pearsonr is not None:
                    _, p_value = scipy_pearsonr(pair["risk"], pair["sentiment_lag"])
                else:
                    z = abs(coef) * np.sqrt(len(pair))
                    p_value = 2 * NormalDist().cdf(-z)
            rows.append(
                {
                    "risk_variable": risk_col,
                    "lagged_sentiment_variable": f"{sentiment_col}_lag{lag}",
                    "n": len(pair),
                    "correlation": coef,
                    "p_value": p_value,
                    "reported": _format_corr(coef, p_value),
                }
            )
    return write_csv(pd.DataFrame(rows), output_csv)


def lagged_correlation_table_from_sources(
    sources: list[dict[str, object]],
    output_csv: Path,
    sentiment_cols: list[str] | None = None,
    lag: int = 1,
) -> Path:
    sentiment_cols = sentiment_cols or SENTIMENT_FEATURE_COLUMNS
    rows = []
    for source in sources:
        df = read_csv_auto(Path(str(source["input_csv"])))
        sample = str(source.get("sample", ""))
        for risk_col in source["risk_cols"]:
            for sentiment_col in sentiment_cols:
                pair = pd.DataFrame(
                    {
                        "risk": pd.to_numeric(df[str(risk_col)], errors="coerce"),
                        "sentiment_lag": pd.to_numeric(df[sentiment_col], errors="coerce").shift(lag),
                    }
                ).dropna()
                if len(pair) < 3 or pair["risk"].nunique() <= 1 or pair["sentiment_lag"].nunique() <= 1:
                    coef, p_value = np.nan, np.nan
                else:
                    coef = float(pair["risk"].corr(pair["sentiment_lag"]))
                    if scipy_pearsonr is not None:
                        _, p_value = scipy_pearsonr(pair["risk"], pair["sentiment_lag"])
                    else:
                        z = abs(coef) * np.sqrt(len(pair))
                        p_value = 2 * NormalDist().cdf(-z)
                rows.append(
                    {
                        "sample": sample,
                        "risk_variable": DISPLAY_NAMES.get(str(risk_col), str(risk_col)),
                        "lagged_sentiment_variable": DISPLAY_NAMES.get(sentiment_col, sentiment_col),
                        "lag": lag,
                        "n": len(pair),
                        "correlation": coef,
                        "p_value": p_value,
                        "reported": _format_corr(coef, p_value),
                    }
                )
    audit = pd.DataFrame(rows)
    write_csv(audit, output_csv.with_name(f"{output_csv.stem}_audit{output_csv.suffix}"))

    column_specification = [
        ("ICBC", "ICBC SR proxy", "SR proxy"),
        ("ICBC", "DY spillover To", "DY spillover To"),
        ("Banking sector", "Banking sector SR proxy", "SR proxy"),
    ]
    columns = pd.MultiIndex.from_tuples(
        [("Index", "")] + [(sample, label) for sample, _, label in column_specification]
    )
    table_rows: list[list[str]] = []
    for feature in SENTIMENT_FEATURE_COLUMNS:
        display_feature = DISPLAY_NAMES[feature]
        values = []
        for sample, risk_variable, _ in column_specification:
            match = audit.loc[
                (audit["sample"] == sample)
                & (audit["risk_variable"] == risk_variable)
                & (audit["lagged_sentiment_variable"] == display_feature),
                "reported",
            ]
            values.append(match.iloc[0] if not match.empty else "")
        table_rows.append([display_feature, *values])
    return write_csv(pd.DataFrame(table_rows, columns=columns), output_csv)


def model_comparison_table(metrics_csv: Path, output_csv: Path) -> Path:
    df = read_csv_auto(metrics_csv)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if "model" in df.columns:
        out = df.copy()
        out["model"] = out["model"].replace(DISPLAY_NAMES).astype(str).str.replace("_pred", "", regex=False)
        reported_models = [
            "Informer",
            "Persistence",
            "AR(1)",
            "ETS(A,N,N)",
            "Ridge-ARX",
            "Random Forest",
            "LSTM",
        ]
        out = out.set_index("model").reindex(reported_models)
        rows = []
        for metric in ["MSE", "MAE"]:
            values = pd.to_numeric(out[metric], errors="coerce")
            informer = values.loc["Informer"]
            best_comparator = values.drop(index="Informer").min()
            improvement = (best_comparator - informer) / best_comparator * 100
            formatted = [
                f"{float(values.loc[model]):.2e}" if metric == "MSE" else f"{float(values.loc[model]):.3f}"
                for model in reported_models
            ]
            rows.append([metric, *formatted, f"{improvement:.2f}%"])
        headers = ["Error term"] + reported_models + ["IMP"]
        return write_csv(pd.DataFrame(rows, columns=headers), output_csv)
    else:
        test = df.loc[df["数据集"] == "test"].copy()
        out = test.groupby(["银行", "情绪"], as_index=False)[["MSE", "MAE", "RMSE", "R2"]].mean()
    return write_csv(out, output_csv)


def dm_test_table(dm_1d_csv: Path, dm_5d_csv: Path, output_csv: Path) -> Path:
    short_names = {
        "Persistence": "Persistence",
        "AR1": "AR(1)",
        "ETS_ANN": "ETS(A,N,N)",
        "Ridge ARX": "Ridge-ARX",
        "Random Forest Regressor": "RF",
        "LSTM": "LSTM",
    }
    model_columns = list(short_names)
    rows: list[list[str]] = []
    for horizon, path in [("1-step-ahead prediction", dm_1d_csv), ("Five-output exercise", dm_5d_csv)]:
        df = read_csv_auto(path).loc[:, lambda frame: ~frame.columns.str.startswith("Unnamed")]
        rows.append([horizon, *([horizon] * len(model_columns))])
        rows.append(["", *[short_names[column] for column in model_columns]])
        for source_criterion, display_criterion in [("MAD", "MAE"), ("MSE", "MSE")]:
            record = df.loc[df["criterion"] == source_criterion]
            if record.empty:
                raise ValueError(f"Missing {source_criterion} DM-test row in {path}")
            rows.append(
                [display_criterion]
                + [str(record.iloc[0][column]).replace(" ", "") for column in model_columns]
            )
    return write_csv(pd.DataFrame(rows), output_csv, header=False)


def bank_error_table(metrics_csv: Path, output_csv: Path) -> Path:
    df = read_csv_auto(metrics_csv)
    required = {"数据集", "银行", "情绪", "MSE", "MAE"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing metrics columns: {sorted(missing)}")
    test = df.loc[df["数据集"] == "test"].copy()
    avg = test.groupby(["银行", "情绪"], as_index=False)[["MSE", "MAE", "RMSE", "R2"]].mean()
    wide = avg.pivot(index="银行", columns="情绪", values=["MSE", "MAE"])
    rows = []
    for bank in avg["银行"].drop_duplicates():
        bank_rows = avg.loc[avg["银行"] == bank]
        for _, row in bank_rows.iterrows():
            rows.append(
                {
                    "银行": bank,
                    "模型设定": row["情绪"],
                    "MSE": row["MSE"],
                    "MAE": row["MAE"],
                    "RMSE": row.get("RMSE", np.nan),
                    "R2": row.get("R2", np.nan),
                }
            )
        if ("MSE", "不带情绪") in wide.columns and ("MSE", "带情绪") in wide.columns and bank in wide.index:
            base_mse = wide.loc[bank, ("MSE", "不带情绪")]
            sen_mse = wide.loc[bank, ("MSE", "带情绪")]
            base_mae = wide.loc[bank, ("MAE", "不带情绪")]
            sen_mae = wide.loc[bank, ("MAE", "带情绪")]
            rows.append(
                {
                    "银行": bank,
                    "模型设定": "误差降低率",
                    "MSE": (base_mse - sen_mse) / base_mse if base_mse else np.nan,
                    "MAE": (base_mae - sen_mae) / base_mae if base_mae else np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                }
            )
    return write_csv(pd.DataFrame(rows), output_csv)


def bank_sentiment_extreme_stats(metrics_csv: Path, output_csv: Path) -> Path:
    df = read_csv_auto(metrics_csv)
    required = {"数据集", "银行", "情绪", "MSE", "MAE", "RMSE", "R2", "MAPE", "MSPE"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing metrics columns: {sorted(missing)}")
    metrics = ["MSE", "MAE", "RMSE", "R2", "MAPE", "MSPE"]
    test = df.loc[df["数据集"] == "test"].copy()
    rows = []
    for bank in test["银行"].dropna().drop_duplicates():
        subset = test.loc[test["银行"] == bank]
        for sentiment_label, selector in [("不带情绪", "worst"), ("带情绪", "best")]:
            side = subset.loc[subset["情绪"] == sentiment_label]
            if side.empty:
                continue
            row = {"银行": bank, "数据集": "test", "情绪": sentiment_label}
            for metric in metrics:
                if metric == "R2":
                    row[metric] = side[metric].min() if selector == "worst" else side[metric].max()
                else:
                    row[metric] = side[metric].max() if selector == "worst" else side[metric].min()
            rows.append(row)
    return write_csv(pd.DataFrame(rows), output_csv)
