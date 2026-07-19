from __future__ import annotations

import argparse
import inspect
import os
import site
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import load_json, resolve_path, run_python_entry, write_json


def _validate_accelerate_compatibility(
    unwrap_parameters: Iterable[str] | None = None,
    accelerate_version: str | None = None,
    transformers_version: str | None = None,
) -> None:
    """Fail before model loading when Transformers and Accelerate APIs do not match."""
    if unwrap_parameters is None:
        try:
            import accelerate
            import transformers
            from accelerate import Accelerator
        except ImportError as exc:
            raise RuntimeError(
                "The full training dependencies are missing. Install "
                "pipeline/requirements_full_training.txt."
            ) from exc
        unwrap_parameters = inspect.signature(Accelerator.unwrap_model).parameters
        accelerate_version = accelerate.__version__
        transformers_version = transformers.__version__
    if "keep_torch_compile" not in set(unwrap_parameters):
        raise RuntimeError(
            "Incompatible Transformers/Accelerate installation: "
            f"transformers={transformers_version or 'unknown'}, "
            f"accelerate={accelerate_version or 'unknown'}. "
            "This training stack requires accelerate==1.3.0; run "
            "`python -m pip install accelerate==1.3.0`."
        )


def _isolate_user_site_packages() -> list[str]:
    """Keep this Conda environment from importing packages from the user site."""
    os.environ["PYTHONNOUSERSITE"] = "1"
    raw_user_sites = site.getusersitepackages()
    user_sites = [raw_user_sites] if isinstance(raw_user_sites, str) else list(raw_user_sites)
    normalized_user_sites = {
        os.path.normcase(os.path.abspath(path)) for path in user_sites if path
    }
    removed: list[str] = []
    retained: list[str] = []
    for entry in sys.path:
        normalized_entry = os.path.normcase(os.path.abspath(entry or os.curdir))
        if normalized_entry in normalized_user_sites:
            removed.append(entry)
        else:
            retained.append(entry)
    sys.path[:] = retained
    return removed


def _normalize_strategy_argument(
    training_args: dict,
    accepted_fields: Iterable[str] | None = None,
) -> dict:
    """Map the renamed Transformers evaluation-strategy argument across versions."""
    normalized = dict(training_args)
    if accepted_fields is None:
        try:
            from transformers import TrainingArguments
        except ImportError as exc:
            raise RuntimeError(
                "Transformers is required for LoRA training. Install the full training "
                "environment before running this script."
            ) from exc
        accepted_fields = TrainingArguments.__dataclass_fields__
    accepted = set(accepted_fields)

    old_name = "evaluation_strategy"
    new_name = "eval_strategy"
    if old_name in normalized and new_name in normalized:
        if normalized[old_name] != normalized[new_name]:
            raise ValueError(
                f"Conflicting values were provided for {old_name!r} and {new_name!r}."
            )
        if old_name in accepted and new_name not in accepted:
            normalized.pop(new_name)
        else:
            normalized.pop(old_name)

    if old_name in normalized and old_name not in accepted and new_name in accepted:
        normalized[new_name] = normalized.pop(old_name)
        print(f"[INFO] Transformers compatibility: mapped {old_name} to {new_name}.")
    elif new_name in normalized and new_name not in accepted and old_name in accepted:
        normalized[old_name] = normalized.pop(new_name)
        print(f"[INFO] Transformers compatibility: mapped {new_name} to {old_name}.")
    return normalized


def _debugger_attached() -> bool:
    return (
        sys.gettrace() is not None
        or os.environ.get("PYCHARM_HOSTED") == "1"
        or any(name == "pydevd" or name.startswith("_pydevd") for name in sys.modules)
    )


def _apply_windows_debug_worker_guard(
    training_args: dict,
    platform_name: str | None = None,
    debugger_attached: bool | None = None,
) -> dict:
    """Avoid pydevd connection failures in spawned Datasets worker processes."""
    normalized = dict(training_args)
    platform_name = os.name if platform_name is None else platform_name
    debugger_attached = _debugger_attached() if debugger_attached is None else debugger_attached
    workers = normalized.get("preprocessing_num_workers")
    if platform_name == "nt" and debugger_attached and workers is not None and int(workers) > 1:
        normalized["preprocessing_num_workers"] = 1
        print(
            "[INFO] Windows debugger detected: preprocessing_num_workers was reduced "
            f"from {workers} to 1 to prevent pydevd multiprocessing failures."
        )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoRA SFT for the sentiment classifier from a JSON config.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "02_lora_sft.json"))
    parser.add_argument("--dry-run", action="store_true", help="Print the command without starting GPU training.")
    args = parser.parse_args()

    removed_user_sites = _isolate_user_site_packages()
    if removed_user_sites:
        print("[INFO] Ignoring user-site packages for environment isolation: " + ", ".join(removed_user_sites))

    cfg = load_json(args.config)
    if cfg.get("cuda_visible_devices") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_visible_devices"])
    if args.dry_run:
        training_args = _normalize_strategy_argument(
            dict(cfg["training_args"]), accepted_fields={"eval_strategy"}
        )
    else:
        _validate_accelerate_compatibility()
        training_args = _normalize_strategy_argument(dict(cfg["training_args"]))
    training_args = _apply_windows_debug_worker_guard(training_args)
    effective_config = resolve_path(cfg.get("effective_config", "outputs/effective_lora_sft_config.json"))
    write_json(effective_config, training_args)
    script = resolve_path(cfg.get("training_script", "libs/sentiment_training/run_clm_sft_with_peft.py"))
    run_python_entry(script, [str(effective_config)], cwd=PROJECT_ROOT, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
