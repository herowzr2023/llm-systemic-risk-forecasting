from __future__ import annotations

import json
import importlib.util
import inspect
import os
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[2]


def resolve_path(raw_path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_json(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return resolved


def read_csv_auto(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    resolved = resolve_path(path)
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(resolved, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(resolved, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, index: bool = False, header: bool = True) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(resolved, index=index, header=header, encoding="utf-8-sig")
        return resolved
    except PermissionError:
        fallback = resolved.with_name(f"{resolved.stem}_new{resolved.suffix}")
        df.to_csv(fallback, index=index, header=header, encoding="utf-8-sig")
        print(f"[WARN] {resolved} is locked. Wrote fallback file: {fallback}")
        return fallback


@contextmanager
def _temporary_script_context(script: Path, argv: list[str], cwd: Path | None):
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    old_path = sys.path[:]
    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    sys.argv = [str(script), *argv]
    if cwd is not None:
        os.chdir(cwd)
    try:
        yield
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
        os.chdir(old_cwd)


def run_python_entry(script: str | Path, argv: list[str], cwd: str | Path | None = None, dry_run: bool = False) -> int:
    resolved_script = resolve_path(script)
    resolved_cwd = resolve_path(cwd) if cwd else None
    print(" ".join([sys.executable, str(resolved_script), *map(str, argv)]))
    if dry_run:
        return 0
    source = resolved_script.read_text(encoding="utf-8", errors="ignore")
    if "def main(" not in source:
        with _temporary_script_context(resolved_script, [str(arg) for arg in argv], resolved_cwd):
            runpy.run_path(str(resolved_script), run_name="__main__")
        return 0
    module_name = f"_pipeline_entry_{resolved_script.stem}_{abs(hash(resolved_script))}"
    spec = importlib.util.spec_from_file_location(module_name, resolved_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Python entry: {resolved_script}")
    module = importlib.util.module_from_spec(spec)
    with _temporary_script_context(resolved_script, [str(arg) for arg in argv], resolved_cwd):
        spec.loader.exec_module(module)
        main_func = getattr(module, "main", None)
        if callable(main_func):
            signature = inspect.signature(main_func)
            if len(signature.parameters) == 0:
                main_func()
            else:
                main_func([str(arg) for arg in argv])
    return 0


def project_python() -> str:
    return sys.executable
