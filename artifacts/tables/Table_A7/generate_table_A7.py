from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
BANKS = ["CNCB", "BOC", "BCM", "CIB", "BOB", "HB", "BON", "NBCB", "ICBC", "PAB", "CCB", "CMB", "CMBC", "SPDB"]
BANK_MAP = {
    "中信银行": "CNCB", "中国银行": "BOC", "交通银行": "BCM", "兴业银行": "CIB",
    "北京银行": "BOB", "华夏银行": "HB", "南京银行": "BON", "宁波银行": "NBCB",
    "工商银行": "ICBC", "平安银行": "PAB", "建设银行": "CCB", "招商银行": "CMB",
    "民生银行": "CMBC", "浦发银行": "SPDB",
}
SPECIFICATION_MAP = {"不带情绪": "without sentiment", "带情绪": "with sentiment"}


def load_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(columns={"银行": "bank", "数据集": "dataset", "情绪": "sentiment_specification"})
    frame["bank"] = frame["bank"].map(BANK_MAP)
    frame["sentiment_specification"] = frame["sentiment_specification"].map(SPECIFICATION_MAP)
    if frame[["bank", "sentiment_specification"]].isna().any().any():
        raise ValueError("Unrecognized bank or sentiment specification in the source CSV")
    # One archived BON row has a blanked MSE value but a valid RMSE. Restore the identity MSE = RMSE^2.
    inconsistent = (pd.to_numeric(frame["MSE"], errors="coerce") - pd.to_numeric(frame["RMSE"], errors="coerce") ** 2).abs() > 1e-6
    frame.loc[inconsistent, "MSE"] = pd.to_numeric(frame.loc[inconsistent, "RMSE"], errors="coerce") ** 2
    return frame


def reduction(without: float, with_sentiment: float) -> str:
    return f"{(without - with_sentiment) / without * 100:.2f}%" if without else "NA"


def block(frame: pd.DataFrame, bank: str) -> list[list[str]]:
    selected = frame.loc[frame["bank"].eq(bank)].set_index("sentiment_specification")
    without = selected.loc["without sentiment"]
    with_sentiment = selected.loc["with sentiment"]
    return [[bank, f"{without.MSE * 1e4:.2f}", f"{without.MAE * 1e2:.2f}"],
            [bank, f"{with_sentiment.MSE * 1e4:.2f}", f"{with_sentiment.MAE * 1e2:.2f}"],
            [bank, reduction(without.MSE, with_sentiment.MSE), reduction(without.MAE, with_sentiment.MAE)]]


def main() -> None:
    frame = load_source(ROOT / "data" / "bank_sentiment_stats_5_out.csv")
    left, right = BANKS[:7], BANKS[7:]
    rows = [["Bank", "MSE (×10⁻⁴)", "MAE (×10⁻²)", "Bank", "MSE (×10⁻⁴)", "MAE (×10⁻²)"]]
    for first, second in zip(left, right):
        for left_row, right_row in zip(block(frame, first), block(frame, second)):
            rows.append([*left_row, *right_row])
    output = ROOT / "output" / "Table_A7.csv"
    output.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, header=False, encoding="utf-8-sig")
    print(output)


if __name__ == "__main__":
    main()
