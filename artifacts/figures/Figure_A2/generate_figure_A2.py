from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODELS = [("Informer", "baseline_pred"), ("Persistence", "Persistence_pred"),
          ("AR(1)", "AR1_pred"), ("ETS", "ETS_ANN_pred"), ("GBR", "GradientBoostingRegressor_pred"),
          ("PatchTST", "PatchTST_pred")]
STYLES = ["--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "x", "+"]


def main() -> None:
    frame = pd.read_csv(ROOT / "data" / "complete_base_5d_predictions.csv")
    frame["date"] = pd.to_datetime(frame["date"])
    figure, axes = plt.subplots(ceil(len(MODELS)/2), 2, figsize=(14, 10), sharex=True)
    for index, (name, column) in enumerate(MODELS):
        axis = np.asarray(axes).ravel()[index]
        axis.plot(frame.date, frame.TRUE, color="black", linewidth=1.3, label="Observed")
        axis.plot(frame.date, frame[column], color="black", linestyle=STYLES[index],
                  marker=MARKERS[index], markevery=max(1, len(frame)//24),
                  markersize=2.4, linewidth=1.0, label="Forecast")
        axis.set_title(name); axis.set_xlabel("Target date"); axis.set_ylabel("SR proxy")
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axis.tick_params(axis="x", rotation=45); axis.legend(frameon=False, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output = ROOT / "output" / "Figure_A2.png"; output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=600, bbox_inches="tight", facecolor="white"); plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
