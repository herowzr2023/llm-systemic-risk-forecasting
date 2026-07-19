from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import load_json, resolve_path
from forecast_data import ForecastDataConfig, build_all_forecast_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bank-level and aggregate SR forecasting datasets.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "05_forecast_data.json"))
    args = parser.parse_args()

    raw = load_json(args.config)
    cfg = ForecastDataConfig(
        covar_csv=resolve_path(raw["covar_csv"]),
        final_stats_csv=resolve_path(raw["final_stats_csv"]),
        bank_mapping_csv=resolve_path(raw["bank_mapping_csv"]),
        price_csv=resolve_path(raw["price_csv"]),
        connectedness_from_csv=resolve_path(raw["connectedness_from_csv"]),
        connectedness_to_csv=resolve_path(raw["connectedness_to_csv"]),
        connectedness_net_csv=resolve_path(raw["connectedness_net_csv"]),
        macro_csv=resolve_path(raw["macro_csv"]),
        credit_csv=resolve_path(raw["credit_csv"]),
        volatility_csv=resolve_path(raw["volatility_csv"]),
        output_dir=resolve_path(raw.get("output_dir", "outputs/forecast_data")),
        target_scale=float(raw.get("target_scale", -100.0)),
    )
    outputs = build_all_forecast_datasets(cfg, build_legacy_system=bool(raw.get("build_legacy_system", True)))
    for path in outputs:
        print(f"wrote: {path}")


if __name__ == "__main__":
    main()
