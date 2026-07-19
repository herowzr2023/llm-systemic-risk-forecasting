from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
MODELS = [
    ("Informer", "baseline"),
    ("Persistence", "Persistence"),
    ("AR(1)", "AR1"),
    ("ETS", "ETS_ANN"),
    ("GBR", "GradientBoostingRegressor"),
    ("PatchTST", "PatchTST"),
]


def main() -> None:
    source = pd.read_csv(ROOT / "data" / "complete_base_1d_compare_metrics.csv").set_index("model")
    values = pd.DataFrame({label: source.loc[source_name] for label, source_name in MODELS}).T
    informer = values.loc["Informer"]
    competitors = values.drop(index="Informer")
    improvement_mse = (competitors["MSE"].min() - informer["MSE"]) / competitors["MSE"].min() * 100
    improvement_mae = (competitors["MAE"].min() - informer["MAE"]) / competitors["MAE"].min() * 100
    labels = [label for label, _ in MODELS]
    rows = [
        ["Error term", *labels, "IMP"],
        ["MSE (×10⁻⁴)", *[f"{values.loc[m, 'MSE'] * 1e4:.2f}" for m in labels], f"{improvement_mse:.2f}%"],
        ["MAE (×10⁻²)", *[f"{values.loc[m, 'MAE'] * 1e2:.2f}" for m in labels], f"{improvement_mae:.2f}%"],
    ]
    output = ROOT / "output" / "Table_3.csv"
    output.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, header=False, encoding="utf-8-sig")
    print(output)


if __name__ == "__main__":
    main()
