from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
INDICATORS = [
    ("Emotional_Skew", "Sentiment direction"),
    ("Total_Count", "Attention-based intensity"),
]


def stats(path: Path, column: str) -> list[str]:
    values = pd.to_numeric(pd.read_csv(path)[column], errors="coerce").dropna()
    return [f"{values.mean():.2f}", f"{values.std():.2f}", f"{values.min():.2f}", f"{values.max():.2f}"]


def main() -> None:
    icbc = ROOT / "data" / "工商银行_final_data_with_sen.csv"
    sector = ROOT / "data" / "ALL_final_data_with_sen_full_predictors.csv"
    rows = [
        ["Index", "ICBC", "ICBC", "ICBC", "ICBC", "Banking sector", "Banking sector", "Banking sector", "Banking sector"],
        ["Index", "Mean", "Standard", "Min", "Max", "Mean", "Standard", "Min", "Max"],
    ]
    for column, label in INDICATORS:
        rows.append([label, *stats(icbc, column), *stats(sector, column)])
    output = ROOT / "output" / "Table_A3.csv"
    output.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, header=False, encoding="utf-8-sig")
    print(output)


if __name__ == "__main__":
    main()
