from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
COMPARATORS = [
    ("Persistence", "Persistence"),
    ("AR(1)", "AR1"),
    ("ETS", "ETS_ANN"),
    ("GBR", "GradientBoostingRegressor"),
    ("PatchTST", "PatchTST"),
]


def statistic(value: object) -> str:
    return str(value).replace(" ", "")


def block(path: Path, title: str) -> list[list[str]]:
    data = pd.read_csv(path).set_index("criterion")
    labels = [label for label, _ in COMPARATORS]
    columns = [column for _, column in COMPARATORS]
    return [
        [title] * (len(COMPARATORS) + 1),
        ["", *labels],
        ["MAE", *[statistic(data.loc["MAD", column]) for column in columns]],
        ["MSE", *[statistic(data.loc["MSE", column]) for column in columns]],
    ]


def main() -> None:
    rows = block(ROOT / "data" / "complete_base_1d_compare_DM_test.csv", "One-day-ahead forecast")
    rows += block(ROOT / "data" / "complete_base_5d_compare_DM_test.csv", "Five-day-ahead forecast")
    output = ROOT / "output" / "Table_5.csv"
    output.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, header=False, encoding="utf-8-sig")
    print(output)


if __name__ == "__main__":
    main()
