"""Task specs: tasks/<task_id>/spec.yaml, validated against tasks/schema.json.

Specs may contain <<FILL placeholders while being drafted; `frozen` rejects
them, and batch creation calls it for every task in the plan.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import jsonschema
import yaml

from .config import REPO_ROOT, substitute
from .prereg import PLACEHOLDER

TASKS_DIR = REPO_ROOT / "tasks"
SCHEMA = TASKS_DIR / "schema.json"


class SpecError(ValueError):
    pass


def task_ids(tasks_dir: Path = TASKS_DIR) -> list[str]:
    return sorted(p.parent.name for p in tasks_dir.glob("*/spec.yaml"))


def load_spec(task_id: str, tasks_dir: Path = TASKS_DIR) -> dict:
    with open(tasks_dir / task_id / "spec.yaml") as fh:
        return yaml.safe_load(fh)


def fills(obj, where: str = "") -> list[str]:
    """Dotted paths of every value that is still a <<FILL placeholder."""
    if isinstance(obj, str):
        return [where] if PLACEHOLDER in obj else []
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in fills(v, f"{where}.{k}" if where else k)]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in fills(v, f"{where}[{i}]")]
    return []


def _filled(v) -> bool:
    return not fills(v)


def validate(spec: dict, task_id: str | None = None) -> list[str]:
    """Schema errors plus cross-field rules. Placeholders are allowed here."""
    schema = json.loads(SCHEMA.read_text())
    errors = [f"{'.'.join(map(str, e.path)) or '<root>'}: {e.message}"
              for e in jsonschema.Draft202012Validator(schema).iter_errors(spec)]
    if errors:
        return errors
    if task_id and spec["id"] != task_id:
        errors.append(f"id {spec['id']!r} does not match directory {task_id!r}")
    data = spec["data"]
    for key in ("dev_inputs", "dev_labels", "test_inputs"):
        if _filled(data[key]):
            errors += [f"data.{key}: {p} must start with {{data_root}}/ or {{task_dir}}/"
                       for p in data[key] if not p.startswith(("{data_root}/", "{task_dir}/"))]
    if _filled(data["truth"]):
        if not data["truth"].startswith("{truth_root}/"):
            errors.append("data.truth must start with {truth_root}/")
        if _filled(data["test_inputs"]) and data["truth"] in data["test_inputs"]:
            errors.append("data.truth must not be staged as a test input")
    budget, auto = spec["budgets"], spec["baselines"]["automl"]["budget"]
    for key in ("wall_time_s", "gpu_hours"):
        if _filled(budget[key]) and _filled(auto[key]) and budget[key] != auto[key]:
            errors.append(f"baselines.automl.budget.{key} ({auto[key]}) must equal budgets.{key} ({budget[key]})")
    return errors


def frozen(spec: dict) -> None:
    left = fills(spec)
    if left:
        raise SpecError(f"task {spec.get('id')!r} has {len(left)} unfilled field(s): {', '.join(left)}")


def resolve(spec: dict, cfg: dict, tasks_dir: Path = TASKS_DIR) -> dict:
    return substitute(spec, {**cfg["paths"], "task_dir": str((tasks_dir / spec["id"]).resolve())})


def ref(name: str):
    """'package.module:attr' -> the object."""
    module, attr = name.split(":")
    return getattr(importlib.import_module(module), attr)
