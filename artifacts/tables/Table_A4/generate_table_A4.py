from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parent
INDICATORS = [
    ("Emotional_Skew", "Sentiment direction"),
    ("Total_Count", "Attention-based intensity"),
]


def correlation(path: Path, sentiment: str, risk: str) -> str:
    frame = pd.read_csv(path, usecols=[sentiment, risk]).apply(pd.to_numeric, errors="coerce")
    frame[sentiment] = frame[sentiment].shift(1)
    pair = frame.dropna()
    value, p_value = pearsonr(pair[sentiment], pair[risk])
    stars = "***" if p_value < .01 else "**" if p_value < .05 else "*" if p_value < .10 else ""
    return f"{value:.3f}{stars}"


def main() -> None:
    icbc = ROOT / "data" / "工商银行_final_data_with_sen.csv"
    sector = ROOT / "data" / "ALL_final_data_with_sen_full_predictors.csv"
    rows = [["Index", "ICBC", "ICBC", "Banking sector"],
            ["Index", "SR proxy", "DY spillover To", "SR proxy"]]
    for column, label in INDICATORS:
        rows.append([
            label,
            correlation(icbc, column, "工商银行_CoVaR"),
            correlation(icbc, column, "risk_spillover"),
            correlation(sector, column, "ALL_CoVaR"),
        ])
    output = ROOT / "output" / "Table_A4.csv"
    output.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, header=False, encoding="utf-8-sig")
    print(output)


if __name__ == "__main__":
    main()
