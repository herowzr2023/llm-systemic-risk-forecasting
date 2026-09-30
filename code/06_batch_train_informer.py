from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import load_json, project_python, resolve_path, run_python_entry


def load_banks(path: str) -> list[str]:
    df = pd.read_csv(resolve_path(path))
    return df["bank"].astype(str).tolist()


SETTING_TEMPLATE = (
    "{model}_{data}_ft{features}_sl{seq_len}_ll{label_len}_pl{pred_len}_dm{d_model}_nh{n_heads}_el{e_layers}_"
    "dl{d_layers}_df{d_ff}_at{attn}_fc{factor}_eb{embed}_dt{distil}_mx{mix}_path{path}_{des}_{itr}"
)


def variants_for_job(job: dict) -> list[dict]:
    if "variants" in job:
        return [dict(variant) for variant in job["variants"]]
    return [
        {
            "name": job.get("name", "default"),
            "sentiment_label": job.get("sentiment_label", "带情绪"),
            "data": job["data"],
            "data_path": job["data_path"],
            "predict_col": job["predict_col"],
            "description": job["description"],
            "args": job.get("args", {}),
        }
    ]


def effective_args(raw: dict, job: dict, variant: dict, bank: str | None, iteration: int) -> dict:
    common = dict(raw["common_args"])
    common.update(job.get("args", {}))
    common.update(variant.get("args", {}))
    common.update(
        {
            "model": common.get("model", "informer"),
            "data": variant["data"],
            "path": variant["data_path"].format(bank=bank or "ALL"),
            "predict_col": variant["predict_col"].format(bank=bank or "ALL"),
            "des": variant["description"].format(bank=bank or "ALL", iteration=iteration),
            "itr": 1,
        }
    )
    return common


def command_for_job(raw: dict, job: dict, variant: dict, bank: str | None, iteration: int) -> list[str]:
    args = effective_args(raw, job, variant, bank, iteration)
    script = resolve_path(raw.get("main_script", "libs/informer/main_informer.py"))
    command = [
        raw.get("python", project_python()),
        str(script),
        "--data",
        args["data"],
        "--path",
        args["path"],
        "--predict_col",
        args["predict_col"],
        "--des",
        args["des"],
        "--itr",
        "1",
    ]
    for key, value in args.items():
        if key in {"data", "path", "predict_col", "des", "itr"}:
            continue
        if key in {"distil", "mix"}:
            if value is False:
                command.append(f"--{key}")
        elif isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        else:
            command.extend([f"--{key}", str(value)])
    return command


def setting_for_job(raw: dict, job: dict, variant: dict, bank: str | None, iteration: int) -> str:
    args = effective_args(raw, job, variant, bank, iteration)
    values = {
        "model": args.get("model", "informer"),
        "data": args["data"],
        "features": args.get("features", "MS"),
        "seq_len": args.get("seq_len", 5),
        "label_len": args.get("label_len", 2),
        "pred_len": args.get("pred_len", 1),
        "d_model": args.get("d_model", 512),
        "n_heads": args.get("n_heads", 8),
        "e_layers": args.get("e_layers", 2),
        "d_layers": args.get("d_layers", 1),
        "d_ff": args.get("d_ff", 2048),
        "attn": args.get("attn", "prob"),
        "factor": args.get("factor", 5),
        "embed": args.get("embed", "timeF"),
        "distil": str(args.get("distil", True)),
        "mix": str(args.get("mix", True)),
        "path": args["path"],
        "des": args["des"],
        "itr": "1",
    }
    setting = SETTING_TEMPLATE.format(**values)
    if args['data'] == 'direct':
        setting += f"_lead{args['lead']}"
    return setting


def run_informer_command(command: list[str], execute: bool) -> None:
    run_python_entry(command[1], command[2:], cwd=PROJECT_ROOT, dry_run=not execute)


def targets_for_job(raw: dict, job: dict) -> list[str | None]:
    if job.get("target_scope") == "banks":
        return load_banks(raw["bank_mapping_csv"])
    return [None]


def collect_metrics(raw: dict, job: dict, output_csv: str | None = None) -> Path:
    rows = []
    results_dir = resolve_path(raw.get("results_dir", "results"))
    for target in targets_for_job(raw, job):
        bank_name = target or "ALL"
        for variant in variants_for_job(job):
            for iteration in range(1, int(job.get("training_times", 1)) + 1):
                setting = setting_for_job(raw, job, variant, target, iteration)
                metrics_path = results_dir / setting / "metrics.csv"
                if not metrics_path.exists():
                    print(f"missing metrics: {metrics_path}")
                    continue
                metrics = pd.read_csv(metrics_path)
                metrics["银行"] = bank_name
                metrics["情绪"] = variant.get("sentiment_label", variant.get("name", ""))
                metrics["迭代次数"] = iteration
                metrics["setting"] = setting
                rows.append(metrics)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    output = resolve_path(output_csv or job.get("metrics_output_csv", f"outputs/06_informer_batch/{job['name']}_metrics.csv"))
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated Informer jobs from a batch config.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "06_informer_batch.json"))
    parser.add_argument("--job", required=True, help="Job name or 'list'.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true", help="Actually start training. Omit to print commands only.")
    parser.add_argument("--collect-metrics", action="store_true", help="Collect metrics.csv files from completed runs.")
    parser.add_argument("--metrics-output", default=None, help="Optional output CSV path for --collect-metrics.")
    args = parser.parse_args()

    raw = load_json(args.config)
    jobs = {job["name"]: job for job in raw["jobs"]}
    if args.job == "list":
        for job in raw["jobs"]:
            print(f"{job['name']}: {job.get('description_text', '')}")
        return
    if args.job not in jobs:
        raise SystemExit(f"Unknown job: {args.job}")

    job = jobs[args.job]
    if args.collect_metrics:
        print(collect_metrics(raw, job, args.metrics_output))
        return

    execute = args.run and not args.dry_run
    for target in targets_for_job(raw, job):
        for iteration in range(1, int(job.get("training_times", 1)) + 1):
            for variant in variants_for_job(job):
                run_informer_command(command_for_job(raw, job, variant, target, iteration), execute=execute)


if __name__ == "__main__":
    main()
