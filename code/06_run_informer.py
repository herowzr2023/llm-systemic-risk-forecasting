from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import load_json, project_python, resolve_path, run_python_entry


def build_command(raw: dict) -> list[str]:
    script = resolve_path(raw.get("main_script", "libs/informer/main_informer.py"))
    args = raw["args"]
    command = [raw.get("python", project_python()), str(script)]
    for key, value in args.items():
        flag = f"--{key}"
        if key in {"distil", "mix"}:
            if value is False:
                command.append(flag)
        elif isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Informer training/evaluation job.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "06_informer_single.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw = load_json(args.config)
    command = build_command(raw)
    run_python_entry(command[1], command[2:], cwd=PROJECT_ROOT, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
