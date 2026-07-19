from __future__ import annotations

import argparse
import importlib.util
import os
import site
import sys
from pathlib import Path


def _isolate_active_environment() -> None:
    """Prevent user-site packages from overriding the active environment."""
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    try:
        user_site = Path(site.getusersitepackages()).resolve()
    except (AttributeError, TypeError):
        return
    sys.path[:] = [
        entry
        for entry in sys.path
        if not entry or not Path(entry).resolve().is_relative_to(user_site)
    ]


_isolate_active_environment()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import load_json, resolve_path


def _load_runner():
    module_path = PROJECT_ROOT / "libs" / "model_comparison" / "lead_specific_comparison.py"
    spec = importlib.util.spec_from_file_location("lead_specific_comparison", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.run_lead_specific_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit selected comparator models and save their forecasts beside the Informer forecast "
            "for one specified forecast lead."
        )
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "07_model_comparison.json"))
    parser.add_argument("--job", required=True, help="Job name or 'list'.")
    parser.add_argument("--models", nargs="+", help="Optional model subset overriding the job configuration.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configuration = load_json(args.config)
    jobs = {job["name"]: job for job in configuration["jobs"]}
    if args.job == "list":
        for job in configuration["jobs"]:
            print(f"{job['name']}: {job.get('description', '')}")
        return
    if args.job not in jobs:
        raise SystemExit(f"Unknown job {args.job!r}; use --job list")

    job = jobs[args.job]
    models = args.models or job.get("models") or configuration["models"]
    print(
        {
            "job": job["name"],
            "baseline_csv": job["baseline_csv"],
            "data_csv": job["data_csv"],
            "output_csv": job["output_csv"],
            "lead": job["lead"],
            "lags": job["lags"],
            "models": models,
        }
    )
    if args.dry_run:
        return

    runner = _load_runner()
    output = runner(
        baseline_csv=resolve_path(job["baseline_csv"]),
        data_csv=resolve_path(job["data_csv"]),
        output_csv=resolve_path(job["output_csv"]),
        lead=int(job["lead"]),
        lags=int(job["lags"]),
        selected_models=list(models),
        config=configuration,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
