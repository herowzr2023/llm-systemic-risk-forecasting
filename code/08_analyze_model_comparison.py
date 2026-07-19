from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import load_json, resolve_path
from dm import analyze_prediction_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute forecast-error metrics and Informer-referenced DM tests."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "08_dm_analysis.json"))
    parser.add_argument("--job", default="all", help="Job name, 'all', or 'list'.")
    args = parser.parse_args()

    configuration = load_json(args.config)
    jobs = {job["name"]: job for job in configuration["jobs"]}
    if args.job == "list":
        for job in configuration["jobs"]:
            print(f"{job['name']}: {job.get('description', '')}")
        return
    if args.job != "all" and args.job not in jobs:
        raise SystemExit(f"Unknown job {args.job!r}; use --job list")

    selected = configuration["jobs"] if args.job == "all" else [jobs[args.job]]
    for job in selected:
        outputs = analyze_prediction_file(
            input_csv=resolve_path(job["input_csv"]),
            metrics_output_csv=resolve_path(job["metrics_output_csv"]),
            dm_output_csv=resolve_path(job["dm_output_csv"]),
            actual_col=job.get("actual_col", "TRUE"),
            baseline_col=job.get("baseline_col", "Informer_pred"),
            date_col=job.get("date_col", "date"),
        )
        print(f"[{job['name']}]")
        for name, path in outputs.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
