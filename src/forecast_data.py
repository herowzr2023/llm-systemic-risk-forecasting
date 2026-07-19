from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from common import read_csv_auto, write_csv


PRICE_INDICATORS = ["市盈率", "市净率", "市销率", "市现率"]
MACRO_FEATURE_COLUMNS = ["LS", "YS", "CS", "VIX", "Market_Return"]
CONNECTEDNESS_COLUMNS = {
    "risk_receiving": "connectedness_from_csv",
    "risk_spillover": "connectedness_to_csv",
    "net_risk_value": "connectedness_net_csv",
}
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
SENTIMENT_FEATURE_COLUMNS = [
    "Total_Count",
    "Emotional_Skew",
    "Weighted_Emotional_Skew_Read",
    "Weighted_Emotional_Skew_Comment",
]


@dataclass(frozen=True)
class ForecastDataConfig:
    covar_csv: Path
    final_stats_csv: Path
    bank_mapping_csv: Path
    price_csv: Path
    connectedness_from_csv: Path
    connectedness_to_csv: Path
    connectedness_net_csv: Path
    macro_csv: Path
    credit_csv: Path
    volatility_csv: Path
    output_dir: Path
    target_scale: float = -100.0


def read_bank_mapping(path: Path) -> pd.DataFrame:
    mapping = read_csv_auto(path)
    required = {"bank_code", "bank"}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"Missing bank-mapping columns: {sorted(missing)}")
    return mapping


def read_covar_as_sr(path: Path, target_scale: float = -100.0) -> pd.DataFrame:
    covar = read_csv_auto(path, parse_dates=[0], index_col=0)
    covar.index = pd.to_datetime(covar.index)
    covar = covar.apply(pd.to_numeric, errors="coerce")
    return target_scale * covar


def _read_price_panel(path: Path) -> pd.DataFrame:
    return read_csv_auto(path, header=[0, 1, 2], parse_dates=[0], index_col=0, low_memory=False)


def read_market_cap(price_csv: Path) -> pd.DataFrame:
    price = _read_price_panel(price_csv)
    market_cap = price.xs("总市值(元)", level=2, axis=1)
    if isinstance(market_cap.columns, pd.MultiIndex):
        market_cap.columns = market_cap.columns.get_level_values(1)
    market_cap = market_cap.loc[:, ~market_cap.columns.astype(str).str.contains("金融指数")]
    return market_cap.apply(pd.to_numeric, errors="coerce")


def _ordered_bank_columns(values: pd.DataFrame, market_cap: pd.DataFrame, mapping: pd.DataFrame) -> list[str]:
    common = values.columns.intersection(market_cap.columns)
    ordered = [bank for bank in mapping["bank"].astype(str).tolist() if bank in common]
    if not ordered:
        raise ValueError("No mapped banks are shared by the value panel and market-cap panel")
    return ordered


def market_cap_weighted_average(values: pd.DataFrame, market_cap: pd.DataFrame, mapping: pd.DataFrame, output_col: str) -> pd.DataFrame:
    values = values.copy()
    values.index = pd.to_datetime(values.index)
    values = values.apply(pd.to_numeric, errors="coerce")
    lagged_cap = market_cap.shift(1)
    ordered = _ordered_bank_columns(values, lagged_cap, mapping)
    dates = values.index.intersection(lagged_cap.index)
    v = values.loc[dates, ordered].astype(float)
    w = lagged_cap.loc[dates, ordered].astype(float)
    weighted = (v * w).sum(axis=1) / w.sum(axis=1)
    out = pd.DataFrame({output_col: weighted}, index=dates)
    out.index.name = "date"
    return out


def read_connectedness(path: Path, bank_name: str, output_col: str) -> pd.DataFrame:
    data = read_csv_auto(path, parse_dates=["date_suffix"])
    if bank_name not in data.columns:
        raise ValueError(f"{bank_name} not found in {path}")
    data = data.set_index("date_suffix")
    data.index = pd.to_datetime(data.index)
    return data[[bank_name]].rename(columns={bank_name: output_col}).apply(pd.to_numeric, errors="coerce")


def read_connectedness_panel(path: Path, mapping: pd.DataFrame) -> pd.DataFrame:
    data = read_csv_auto(path, parse_dates=["date_suffix"])
    data = data.set_index("date_suffix")
    data.index = pd.to_datetime(data.index)
    bank_cols = [bank for bank in mapping["bank"].astype(str).tolist() if bank in data.columns]
    if not bank_cols:
        raise ValueError(f"No mapped bank columns found in {path}")
    return data[bank_cols].apply(pd.to_numeric, errors="coerce")


def read_price_indicators(price_csv: Path, bank_name: str) -> pd.DataFrame:
    price = _read_price_panel(price_csv)
    selected = [col for col in price.columns if col[1] == bank_name and col[2] in PRICE_INDICATORS]
    if not selected:
        raise ValueError(f"No valuation indicators found for {bank_name}")
    out = price[selected].copy()
    out.columns = [f"{bank_name}_{col[2]}" for col in selected]
    return out.apply(pd.to_numeric, errors="coerce")


def read_price_indicator_panel(price_csv: Path, indicator: str, mapping: pd.DataFrame) -> pd.DataFrame:
    price = _read_price_panel(price_csv)
    out = price.xs(indicator, level=2, axis=1)
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(1)
    bank_cols = [bank for bank in mapping["bank"].astype(str).tolist() if bank in out.columns]
    if not bank_cols:
        raise ValueError(f"No mapped bank columns found for {indicator}")
    return out[bank_cols].apply(pd.to_numeric, errors="coerce")


def append_macro_features(frame: pd.DataFrame, macro_csv: Path, credit_csv: Path, volatility_csv: Path) -> pd.DataFrame:
    out = frame.copy()
    macro = read_csv_auto(macro_csv, parse_dates=[0], index_col=0).dropna(how="all")
    credit = read_csv_auto(credit_csv, parse_dates=[0], index_col=0).dropna(how="all")
    volatility = read_csv_auto(volatility_csv, parse_dates=[0], index_col=0).dropna(how="all")
    imported_columns: list[str] = []
    for source in (macro, credit, volatility):
        source.index = pd.to_datetime(source.index)
        imported_columns.extend(str(column) for column in source.columns if column not in out.columns)
        out = out.merge(source, left_index=True, right_index=True, how="left")

    out.index = pd.to_datetime(out.index)
    # Causal availability rule: at each forecast date, carry forward only the
    # latest external observation already released.  PCHIP/time interpolation
    # can use a later observation on both sides of a gap and therefore leaks
    # future information into an earlier forecast origin.  Do not backfill the
    # initial unavailable segment; downstream dropna() removes it.
    if imported_columns:
        out[imported_columns] = out[imported_columns].ffill()

    out["LS"] = out["银行间同业拆借加权利率:3个月"] - out["中债国债到期收益率:3月"]
    out["YS"] = out["中债国债到期收益率:10年"] - out["中债国债到期收益率:6月"]
    out["CS"] = out["中债国债到期收益率:10年"] - out["中债企业债到期收益率(AAA):10年"]
    out["VIX"] = out["Volatility"]
    out["Market_Return"] = out["Log_Return"]
    return out


def build_bank_without_sentiment(bank_name: str, cfg: ForecastDataConfig) -> pd.DataFrame:
    sr = read_covar_as_sr(cfg.covar_csv, cfg.target_scale)
    target_col = f"{bank_name}_CoVaR"
    if target_col not in sr.columns:
        raise ValueError(f"{target_col} not found in {cfg.covar_csv}")

    merged = sr[[target_col]].copy()
    for col, attr in CONNECTEDNESS_COLUMNS.items():
        merged = merged.merge(read_connectedness(getattr(cfg, attr), bank_name, col), left_index=True, right_index=True, how="left")
    merged = merged.merge(read_price_indicators(cfg.price_csv, bank_name), left_index=True, right_index=True, how="left")
    merged = append_macro_features(merged, cfg.macro_csv, cfg.credit_csv, cfg.volatility_csv)

    price_cols = [c for c in merged.columns if c.startswith(f"{bank_name}_") and c != target_col]
    keep = [target_col, *CONNECTEDNESS_COLUMNS.keys(), *price_cols, *MACRO_FEATURE_COLUMNS]
    final = merged[keep].dropna()
    final.index.name = "date"
    return final


def build_system_without_sentiment(cfg: ForecastDataConfig, include_full_predictors: bool = True) -> pd.DataFrame:
    mapping = read_bank_mapping(cfg.bank_mapping_csv)
    market_cap = read_market_cap(cfg.price_csv)
    sr = read_covar_as_sr(cfg.covar_csv, cfg.target_scale)
    sr.columns = [str(c).replace("_CoVaR", "") for c in sr.columns]
    merged = market_cap_weighted_average(sr, market_cap, mapping, "ALL_CoVaR")

    if include_full_predictors:
        for col, attr in CONNECTEDNESS_COLUMNS.items():
            panel = read_connectedness_panel(getattr(cfg, attr), mapping)
            merged = merged.merge(market_cap_weighted_average(panel, market_cap, mapping, col), left_index=True, right_index=True, how="left")
        for indicator in PRICE_INDICATORS:
            panel = read_price_indicator_panel(cfg.price_csv, indicator, mapping)
            merged = merged.merge(
                market_cap_weighted_average(panel, market_cap, mapping, f"ALL_{indicator}"),
                left_index=True,
                right_index=True,
                how="left",
            )
        merged = append_macro_features(merged, cfg.macro_csv, cfg.credit_csv, cfg.volatility_csv)
        keep = [
            "ALL_CoVaR",
            *CONNECTEDNESS_COLUMNS.keys(),
            *[f"ALL_{indicator}" for indicator in PRICE_INDICATORS],
            *MACRO_FEATURE_COLUMNS,
        ]
        merged = merged[keep]

    final = merged.dropna()
    final.index.name = "date"
    return final


def add_sentiment_features(final_stats: pd.DataFrame) -> pd.DataFrame:
    out = final_stats.copy()
    out["Total_Count"] = out["Positive_Count"] + out["Neutral_Count"] + out["Negative_Count"]
    out["Emotional_Skew"] = np.log((out["Negative_Count"] + 1) / (out["Positive_Count"] + 1))
    out["Weighted_Emotional_Skew_Read"] = np.log((out["Negative_read_num"] + 1) / (out["Positive_read_num"] + 1))
    out["Weighted_Emotional_Skew_Comment"] = np.log(
        (out["Negative_comments_num"] + 1) / (out["Positive_comments_num"] + 1)
    )
    return out


def sentiment_for_bank(final_stats_csv: Path, mapping: pd.DataFrame, bank_name: str | None) -> pd.DataFrame:
    stats = read_csv_auto(final_stats_csv)
    required = {"bank_code", "pub_day", *SENTIMENT_RAW_COLUMNS}
    missing = required.difference(stats.columns)
    if missing:
        raise ValueError(f"Missing final-stats columns: {sorted(missing)}")

    if bank_name is None or bank_name.upper() == "ALL":
        grouped = stats.groupby("pub_day", as_index=False)[SENTIMENT_RAW_COLUMNS].sum()
        sentiment = add_sentiment_features(grouped)
    else:
        rows = mapping.loc[mapping["bank"] == bank_name, "bank_code"]
        if rows.empty:
            raise ValueError(f"{bank_name} not found in bank mapping")
        bank_code = rows.iloc[0]
        sentiment = add_sentiment_features(stats.loc[stats["bank_code"] == bank_code].copy())

    sentiment = sentiment.drop(columns=[c for c in SENTIMENT_RAW_COLUMNS if c in sentiment.columns], errors="ignore")
    sentiment = sentiment.rename(columns={"pub_day": "date"})
    sentiment["date"] = pd.to_datetime(sentiment["date"])
    sentiment = sentiment.drop(columns=["bank_code"], errors="ignore")
    return sentiment[["date", *SENTIMENT_FEATURE_COLUMNS]]


def merge_sentiment(risk_frame: pd.DataFrame, final_stats_csv: Path, mapping: pd.DataFrame, bank_name: str | None) -> pd.DataFrame:
    risk = risk_frame.reset_index().rename(columns={risk_frame.index.name or "index": "date"})
    risk["date"] = pd.to_datetime(risk["date"])
    sentiment = sentiment_for_bank(final_stats_csv, mapping, bank_name)
    return risk.merge(sentiment, on="date", how="left").dropna()


def build_all_forecast_datasets(cfg: ForecastDataConfig, build_legacy_system: bool = True) -> list[Path]:
    mapping = read_bank_mapping(cfg.bank_mapping_csv)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for bank in mapping["bank"].astype(str).tolist():
        no_sen = build_bank_without_sentiment(bank, cfg)
        no_sen_path = write_csv(no_sen.reset_index(), cfg.output_dir / f"{bank}_final_data.csv")
        with_sen = merge_sentiment(no_sen, cfg.final_stats_csv, mapping, bank)
        with_sen_path = write_csv(with_sen, cfg.output_dir / f"{bank}_final_data_with_sen.csv")
        written.extend([no_sen_path, with_sen_path])

    system_full = build_system_without_sentiment(cfg, include_full_predictors=True)
    system_full_path = write_csv(system_full.reset_index(), cfg.output_dir / "ALL_final_data_full_predictors.csv")
    system_full_sen_path = write_csv(
        merge_sentiment(system_full, cfg.final_stats_csv, mapping, "ALL"),
        cfg.output_dir / "ALL_final_data_with_sen_full_predictors.csv",
    )
    written.extend([system_full_path, system_full_sen_path])

    if build_legacy_system:
        system_legacy = build_system_without_sentiment(cfg, include_full_predictors=False)
        system_legacy_path = write_csv(system_legacy.reset_index(), cfg.output_dir / "ALL_final_data.csv")
        system_legacy_sen_path = write_csv(
            merge_sentiment(system_legacy, cfg.final_stats_csv, mapping, "ALL"),
            cfg.output_dir / "ALL_final_data_with_sen.csv",
        )
        written.extend([system_legacy_path, system_legacy_sen_path])
    return written


def selected_bank_names(mapping_csv: Path, include: Iterable[str] | None = None) -> list[str]:
    mapping = read_bank_mapping(mapping_csv)
    banks = mapping["bank"].astype(str).tolist()
    if include:
        requested = set(include)
        banks = [bank for bank in banks if bank in requested]
    return banks
