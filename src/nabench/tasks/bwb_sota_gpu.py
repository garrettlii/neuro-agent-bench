"""The GPU SOTA rerun's rules (D-018, D-019): its 4 h limit, merging seeds run in parallel, and the check of the
merged provenance on import; and the pieces of the exploratory train+val variant (D-022), which never merges.

Kept apart from bwb_neuropixels_wheel, which batch b-20261028-044630-f61b imports while it runs: only the GPU job
(scripts/sota_gpu_job.py, from the export) and scripts/sota_gpu.py use this module.
"""
from __future__ import annotations

import math

LIMIT_S = 4 * 3600  # the whole job, from the first seed starting to the last seed finishing (D-019)
ALL_SEEDS = (1, 2, 3, 4, 5)  # indices into EVAL_SEEDS, as in seeds/seed<k>.csv


class TimeUp(Exception):
    """Raised by a fit once the limit has passed."""


class Stop(Exception):
    """Raised by stop_after's stand-in once its epochs are done, with the predictions after the last one."""

    def __init__(self, pred):
        super().__init__()
        self.pred = pred


def best_epoch(scores) -> int | None:
    """The epoch whose weights poyo_finetune keeps: the first with the highest finite val R^2."""
    best, at = float("-inf"), None
    for epoch, s in enumerate(scores):
        if math.isfinite(s) and s > best:
            best, at = s, epoch
    return at


def stop_after(epochs: int):
    """A stand-in for bwb_neuropixels_wheel.pooled_r2 in the exploratory train+val refit (D-022), where
    poyo_finetune gets the test trials in its validation slot: each call sees the test predictions after one more
    epoch, reports a rising score so that early stopping never starts, and raises Stop with the predictions once
    `epochs` epochs have run."""
    calls = []

    def r2(pred, y):
        calls.append(None)
        if len(calls) == epochs:
            raise Stop(pred)
        return float(len(calls))

    return r2


def part_problems(parts: list[dict], export_manifest_sha256: str) -> list[str]:
    """What stops a set of per-seed job outputs (part.json each) from merging into one 5-seed result: parts from
    another export or other inputs, or seeds missing or run more than once."""
    problems = [f"part {i}: made from another export" for i, p in enumerate(parts, 1)
                if p.get("export_manifest_sha256") != export_manifest_sha256]
    problems += [f"part {i}: the exploratory train+val variant (D-022), never a SOTA result" for i, p in enumerate(parts, 1)
                 if p.get("variant")]
    if len({repr(sorted(p.get("inputs_sha256", {}).items())) for p in parts}) > 1:
        problems.append("parts read different inputs")
    run = [int(k) for p in parts for k in p.get("seeds", {})]
    if sorted(run) != list(ALL_SEEDS):
        problems.append(f"seeds run {sorted(run)}, not each of {list(ALL_SEEDS)} once")
    return problems


def merge_provenance(parts: list[dict]) -> dict:
    """One provenance record for the merged result: each seed's GPU, precision, start and end, and the job's
    overall wall time from the first seed's start to the last seed's end."""
    per_seed = {k: s for p in parts for k, s in p["seeds"].items()}
    start, end = min(s["start_unix"] for s in per_seed.values()), max(s["end_unix"] for s in per_seed.values())
    return {"inputs_sha256": parts[0]["inputs_sha256"], "export_manifest_sha256": parts[0]["export_manifest_sha256"],
            "seeds": [per_seed[str(k)]["eval_seed"] for k in ALL_SEEDS],
            "per_seed": {str(k): per_seed[str(k)] for k in ALL_SEEDS},
            "job_start_unix": start, "job_end_unix": end, "wall_s": round(end - start, 1), "limit_s": LIMIT_S,
            "software": [{k: p.get(k) for k in ("torch", "cuda", "python", "selection")} for p in parts]}


def provenance_problems(prov: dict, eval_seeds) -> list[str]:
    """What in a merged provenance.json breaks D-018 or D-019: a seed missing, without its GPU model or times, in a
    precision other than bf16 where its GPU supports it and fp32 otherwise, or a job longer than 4 h from the first
    seed's start to the last seed's end (recomputed from the seeds' times)."""
    per_seed = prov.get("per_seed") if isinstance(prov.get("per_seed"), dict) else {}
    if sorted(per_seed) != [str(k) for k in ALL_SEEDS] or prov.get("seeds") != list(eval_seeds):
        return [f"not one result per seed for all {len(ALL_SEEDS)} eval seeds"]
    problems = []
    for k, s in sorted(per_seed.items()):
        if s.get("eval_seed") != eval_seeds[int(k) - 1]:
            problems.append(f"seed {k}: eval seed {s.get('eval_seed')}, not {eval_seeds[int(k) - 1]}")
        if not isinstance(s.get("device"), str) or not s["device"].strip():
            problems.append(f"seed {k}: no GPU model recorded")
        if not isinstance(s.get("bf16_supported"), bool) or s.get("precision") != ("bf16" if s["bf16_supported"] else "fp32"):
            problems.append(f"seed {k}: precision is not bf16 on a GPU that supports it, or fp32 on one that does not")
        if not all(isinstance(s.get(t), (int, float)) for t in ("start_unix", "end_unix")) or s["start_unix"] >= s["end_unix"]:
            problems.append(f"seed {k}: no valid start and end time")
    if problems:
        return problems
    wall = max(s["end_unix"] for s in per_seed.values()) - min(s["start_unix"] for s in per_seed.values())
    if prov.get("limit_s") != LIMIT_S or wall > LIMIT_S:
        problems.append(f"the job took {wall:.0f} s from the first seed's start to the last seed's end, over the "
                        f"{LIMIT_S} s limit; under D-019 nothing is imported")
    if not isinstance(prov.get("wall_s"), (int, float)) or abs(prov["wall_s"] - wall) > 1:
        problems.append("recorded wall time does not match the seeds' times")
    return problems
