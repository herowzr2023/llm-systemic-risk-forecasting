from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SERIES = [
    ("Emotional_Skew", "Sentiment direction"),
    ("Total_Count", "Attention-based intensity"),
    ("Weighted_Emotional_Skew_Read", "Exposure-based intensity"),
    ("Weighted_Emotional_Skew_Comment", "Engagement-based intensity"),
    ("工商银行_CoVaR", "SR proxy"),
    ("risk_spillover", "DY spillover To"),
]
STYLES = ["-", "--", "-.", ":", "-", "--"]
MARKERS = ["o", "s", "^", "D", "x", "+"]


def main() -> None:
    frame = pd.read_csv(ROOT / "data" / "工商银行_final_data_with_sen.csv")
    frame["date"] = pd.to_datetime(frame["date"])
    values = frame.set_index("date")[[c for c, _ in SERIES]].apply(pd.to_numeric, errors="coerce")
    values = ((values - values.mean()) / values.std()).resample("ME").mean()
    plt.rcParams.update({"font.family": "DejaVu Serif", "axes.unicode_minus": False})
    figure, axes = plt.subplots(3, 2, figsize=(13.5, 9.4), sharex=True)
    for index, (column, label) in enumerate(SERIES):
        axis = axes.ravel()[index]
        axis.plot(values.index, values[column], color="black", linestyle=STYLES[index],
                  marker=MARKERS[index], markevery=max(1, len(values)//20),
                  markersize=2.5, linewidth=1.1)
        axis.set_title(label)
        axis.set_xlabel("Date")
        axis.set_ylabel("Standardized monthly mean")
        axis.xaxis.set_major_locator(mdates.YearLocator(2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output = ROOT / "output" / "Figure_2.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
