"""Prediction files and per-case metrics.

Predictions are CSV files with a header `case_id,prediction`. They are written by
the agent, so they are parsed defensively: size-capped, unknown and duplicate case
ids are counted and dropped, and a missing case scores as the worst value.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

MAX_PREDICTION_BYTES = 50 << 20
csv.field_size_limit(MAX_PREDICTION_BYTES)  # a case's value can be a long trace (bwb: 50 frames per test trial); the default is 128 KiB


def read_cases(path: str | Path, column: str) -> dict[str, str]:
    """Trusted two-column CSV (truth, case lists) -> {case_id: value}."""
    with open(path, newline="") as fh:
        return {row["case_id"]: row[column] for row in csv.DictReader(fh)}


def write_cases(path: str | Path, cases: dict, column: str = "prediction") -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", column])
        w.writerows(cases.items())


def read_predictions(path: str | Path | None, case_ids) -> tuple[dict[str, str], dict]:
    """Agent-written predictions -> ({case_id: prediction}, parse notes)."""
    notes = {"file": path is not None and Path(path).exists(), "unknown": 0, "duplicate": 0, "error": None}
    if not notes["file"]:
        return {}, notes
    if Path(path).stat().st_size > MAX_PREDICTION_BYTES:
        notes["error"] = "file too large"
        return {}, notes
    known, preds, seen = set(case_ids), {}, set()
    try:
        reader = csv.DictReader(io.StringIO(Path(path).read_text(errors="replace")))
        if not reader.fieldnames or {"case_id", "prediction"} - set(reader.fieldnames):
            notes["error"] = "header must be case_id,prediction"
            return {}, notes
        for row in reader:
            cid = row["case_id"]
            if cid not in known:
                notes["unknown"] += 1
            elif cid in seen:
                notes["duplicate"] += 1
                preds.pop(cid, None)  # ambiguous: a duplicated case counts as missing
            else:
                seen.add(cid)
                preds[cid] = (row["prediction"] or "").strip()
    except csv.Error as exc:
        notes["error"] = f"csv: {exc}"
        return {}, notes
    return preds, notes


def accuracy(pred: dict[str, str], truth: dict[str, str]) -> np.ndarray:
    """1 if the predicted label equals the true label, else 0 (missing = 0); in truth order."""
    return np.array([float(pred.get(cid) == label) for cid, label in truth.items()])
