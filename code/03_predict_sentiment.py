from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import load_json, project_python, resolve_path, run_python_entry


def iter_input_files(patterns: list[str], allow_empty: bool = False) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(p) for p in sorted(glob(str(resolve_path(pattern)))))
    if not files and not allow_empty:
        raise FileNotFoundError(f"No input files matched: {patterns}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sentiment inference for one or more CSV files.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "03_inference.json"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without loading the LLM.")
    args = parser.parse_args()

    cfg = load_json(args.config)
    script = resolve_path(cfg.get("inference_script", "libs/sentiment_inference/inference.py"))
    output_dir = resolve_path(cfg.get("output_dir", "outputs/predicted"))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = iter_input_files(cfg["input_patterns"], allow_empty=args.dry_run)
    if args.dry_run and not input_files:
        print("[DRY-RUN] No forum files are bundled. Expected input patterns:")
        for pattern in cfg["input_patterns"]:
            print(f"  {resolve_path(pattern)}")
        return

    for input_file in input_files:
        command = [
            cfg.get("python", project_python()),
            str(script),
            "--base_model",
            str(resolve_path(cfg["model_path"])),
            "--data_file",
            str(input_file),
            "--content_col",
            cfg.get("content_col", "title"),
            "--target_col",
            cfg.get("target_col", "sen"),
            "--gpus",
            str(cfg.get("gpus", "0")),
            "--output_dir",
            str(output_dir),
        ]
        if cfg.get("with_prompt", True):
            command.append("--with_prompt")
        if cfg.get("load_in_4bit", False):
            command.append("--load_in_4bit")
        if cfg.get("load_in_8bit", False):
            command.append("--load_in_8bit")
        run_python_entry(command[1], command[2:], cwd=PROJECT_ROOT, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
