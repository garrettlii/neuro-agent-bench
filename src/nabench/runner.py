"""Batch creation, workspace staging, agent and baseline runs, and submission.

Each run gets a fresh workspace under runs_root/<batch>/<run_id>/a<attempt>/
holding only TASK.md, the dev data, the test inputs, the condition's tools and
an empty submission/ folder. The agent runs there in the OS sandbox with a hard
wall-time limit; logs stay outside the workspace. Whatever is at
submission/predictions.csv when the agent exits or is stopped is copied to the
scorer's area under truth_root, up to the task's submission limit.
"""
from __future__ import annotations

import hashlib
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from . import agentuser, netprobe, sandbox
from .backends import instruction_files, make_backend, parse_usage
from .batch import (append_jsonl, batch_dirs, check_ledger, checksums, new_batch_id, now, read_json,
                    read_jsonl, record_batch, sha256_file, write_json)
from .config import config_hash
from .metrics import MAX_PREDICTION_BYTES, write_cases
from .spec import TASKS_DIR, fills, frozen, load_spec, ref, resolve, validate

PREDICTIONS = Path("submission") / "predictions.csv"
BASELINES = ("simple", "sota", "automl")
BASELINE_CONDITION = {"mode": "program", "tools": "generic", "scrambled": False}


class RunError(RuntimeError):
    pass


class NotReady(RuntimeError):
    """A baseline or reference whose precomputed inputs are not in place yet: skipped, retried on the next pass."""


def run_id(task: str, condition: str, model: str, rep: int) -> str:
    return f"{task}__{condition}__{model}__r{rep}"


def expand_plan(plan: dict) -> list[dict]:
    return [{"task": t, "condition": c, "model": m, "rep": r, "run_id": run_id(t, c, m, r)}
            for block in plan["blocks"] for t in block["tasks"] for c in block["conditions"]
            for m in block["models"] for r in range(1, block.get("reps", 1) + 1)]


def _staged_sources(spec: dict, tools: bool) -> list[tuple[str, Path]]:
    """(workspace-relative destination, source) for everything a run may see."""
    out = [(f"data/dev/{Path(p).name}", Path(p)) for p in spec["data"]["dev_inputs"] + spec["data"]["dev_labels"]]
    out += [(f"data/test/{Path(p).name}", Path(p)) for p in spec["data"]["test_inputs"]]
    if tools:
        out += [(f"tools/{t['name']}/{Path(p).name}", Path(p)) for t in spec["tools"]["trained_models"] for p in t["files"]]
    return out


def create_batch(cfg: dict, plan: dict, prereg: dict | None, reason: str | None = None,
                 tasks_dir: Path = TASKS_DIR) -> str:
    """prereg None makes a smoke batch, allowed only when every task is synthetic."""
    runs = expand_plan(plan)
    models = {m["name"]: m for m in cfg["models"]}
    specs = {}
    for t in sorted({r["task"] for r in runs}):
        raw = load_spec(t, tasks_dir)
        errors = validate(raw, t)
        if errors:
            raise RunError(f"{t}: invalid spec: {errors}")
        frozen(raw)
        if prereg is None and not raw.get("synthetic"):
            raise RunError(f"{t} is not synthetic; real tasks need a pre-registration")
        specs[t] = resolve(raw, cfg, tasks_dir)
    for r in runs:
        if r["condition"] not in cfg["conditions"]:
            raise RunError(f"unknown condition {r['condition']}")
        mcfg = models.get(r["model"], {})
        if not mcfg.get("enabled"):
            raise RunError(f"model {r['model']} is not configured or not enabled")
        if fills(mcfg):
            raise RunError(f"model {r['model']} has unfilled config fields: {fills(mcfg)}")
        if cfg["conditions"][r["condition"]]["mode"] == "native" and mcfg["backend"] == "cli" and "command_native" not in mcfg:
            raise RunError(f"model {r['model']} has no command_native for native mode")
    for (t, c, m), n in _count(runs).items():
        if n > specs[t]["budgets"]["submission_limit"]:
            raise RunError(f"{n} reps of {t}/{c}/{m} exceed the submission limit")
    check_ledger(cfg, prereg, reason)
    versions = {}
    for name in sorted({r["model"] for r in runs}):
        backend = make_backend(models[name], cfg=cfg)
        versions[name] = {"version": backend.version(), **backend.describe()}
        if prereg is not None and not netprobe.passed(cfg, name, versions[name]["version"], backend.describe()):
            raise RunError(f"model {name} has no passing network probe for version {versions[name]['version']!r} "
                           "and its current command; run scripts/probe_network.py first")

    batch_id = new_batch_id("smoke" if prereg is None else "b")
    created = now()
    run_dir, truth_dir = batch_dirs(cfg, batch_id, create=True)
    manifest = {
        "batch_id": batch_id, "created_at": created, "smoke": prereg is None,
        "prereg": prereg and {k: v for k, v in prereg.items() if k not in ("plan", "predictions")},
        "plan": plan, "predictions": prereg and prereg["predictions"], "runs": runs,
        "config_hash": config_hash(cfg), "config": cfg, "specs": specs, "models": versions,
        "spec_sha256": {t: sha256_file(tasks_dir / t / "spec.yaml") for t in specs},
        "inputs_sha256": {t: checksums(src for _, src in _staged_sources(s, tools=True)) for t, s in specs.items()},
    }
    write_json(run_dir / "manifest.json", manifest)
    truth_manifest = {**manifest, "truth_sha256": {t: sha256_file(s["data"]["truth"]) for t, s in specs.items()}}
    write_json(truth_dir / "manifest.json", truth_manifest)
    (truth_dir / "submissions").mkdir()
    (truth_dir / "scores").mkdir()
    record_batch(cfg, batch_id, created, prereg, reason)
    return batch_id


def _count(runs):
    out = {}
    for r in runs:
        key = (r["task"], r["condition"], r["model"])
        out[key] = out.get(key, 0) + 1
    return out


def task_text(spec: dict, cfg: dict, cond: dict) -> str:
    lines = [f"# Task\n\n{spec['question'].strip()}\n", "## Files\n",
             "- `data/dev/`: development inputs and their labels.",
             "- `data/test/`: test inputs. Test labels are not available."]
    if cond["tools"] == "all" and spec["tools"]["trained_models"]:
        lines.append("- `tools/`: trained models you may call:")
        lines += [f"  - `{t['name']}`: run `{t['call']}`. {t['interface'].strip()}" for t in spec["tools"]["trained_models"]]
    lines += ["", f"Generic software available: {', '.join(spec['tools']['generic'])}. There is no network access.", "",
              "## Output\n", f"Write `{PREDICTIONS}` as CSV with the header `case_id,prediction` and one row per test case. "
              f"{spec['prediction_format'].strip()}", "",
              f"## Working mode\n\n{cfg['modes'][cond['mode']].strip()}", ""]
    return "\n".join(lines)


def stage(spec: dict, cfg: dict, cond: dict, workspace: Path, seed: int, expected: dict | None = None) -> None:
    if workspace.exists():
        raise RunError(f"{workspace} already exists; every run gets a fresh workspace")
    for dest, src in _staged_sources(spec, tools=cond["tools"] == "all"):
        if expected is not None and expected.get(str(src)) != sha256_file(src):
            raise RunError(f"{src} changed since the batch was created")
        (workspace / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, workspace / dest)
    (workspace / "submission").mkdir(parents=True)
    (workspace / "TASK.md").write_text(task_text(spec, cfg, cond))
    if cond.get("scrambled"):
        ref(spec["scramble"]["function"])(workspace / "data", np.random.default_rng(seed), spec)


def submit(truth_dir: Path, run: dict, workspace: Path, limit: int, extra: dict | None = None) -> dict:
    """Copy the predictions file to the scorer's area. Raises once the limit is used up."""
    log = truth_dir / "submissions.jsonl"
    used = [s for s in read_jsonl(log)
            if (s["task"], s["condition"], s["model"]) == (run["task"], run["condition"], run["model"])]
    if len(used) >= limit:
        raise RunError(f"submission limit ({limit}) reached for {run['task']}/{run['condition']}/{run['model']}")
    src, dest = workspace / PREDICTIONS, truth_dir / "submissions" / f"{run['run_id']}.csv"
    rec = {**{k: run[k] for k in ("run_id", "task", "condition", "model", "rep")}, "at": now(), **(extra or {})}
    # the agent controls the workspace: refuse links that would make this copy read outside it
    inside = src.resolve().is_relative_to(workspace.resolve()) and not src.is_symlink()
    try:
        if not (inside and src.is_file() and src.stat().st_size <= MAX_PREDICTION_BYTES):
            raise FileNotFoundError
        shutil.copyfile(src, dest)
        rec.update(file=True, sha256=sha256_file(dest), bytes=dest.stat().st_size)
    except OSError:  # an agent running as another user can also make the file unreadable
        rec.update(file=False, reason="missing, unreadable, symlink or too large")
    append_jsonl(log, rec)
    return rec


def _seed(plan_seed: int, rid: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{plan_seed}:{rid}".encode()).digest()[:8], "little")


def run_baselines(cfg: dict, batch_id: str) -> None:
    run_dir, truth_dir = batch_dirs(cfg, batch_id)
    manifest = read_json(run_dir / "manifest.json")
    done = {s["run_id"] for s in read_jsonl(truth_dir / "submissions.jsonl")}
    for task, spec in manifest["specs"].items():
        baseline_ws = {k: run_dir / run_id(task, "baseline", k, 1) / "a1" / "workspace" for k in BASELINES}
        jobs = [("baseline", k, spec["baselines"][k]) for k in BASELINES]
        jobs += [("reference", r["name"], r) for r in spec.get("references", [])]  # after the baselines they may read
        for cond, name, entry in jobs:
            run = {"task": task, "condition": cond, "model": name, "rep": 1, "run_id": run_id(task, cond, name, 1)}
            if run["run_id"] in done:
                continue
            ws = run_dir / run["run_id"] / "a1" / "workspace"
            if ws.exists():
                shutil.rmtree(ws.parent)  # an interrupted baseline left no submission; baselines are deterministic
            stage(spec, cfg, BASELINE_CONDITION, ws, seed=0, expected=manifest["inputs_sha256"][task])
            budget = entry.get("budget", spec["budgets"])
            fn = ref(entry["function"])
            try:
                preds = fn(ws, budget) if cond == "baseline" else fn(ws, budget, baseline_ws)
            except NotReady as e:  # the agent runs go ahead; the report stays interim until it is in place
                print(f"{run['run_id']}: skipped this pass: {e}")
                shutil.rmtree(ws.parent)
                continue
            write_cases(ws / PREDICTIONS, preds)
            submit(truth_dir, run, ws, limit=1)


def _out_of_attempts(truth_dir: Path, run: dict, attempts: int) -> None:
    """No exclusions: an empty submission scores every case as missing (worst value), so the report can become final."""
    append_jsonl(truth_dir / "submissions.jsonl", {**{k: run[k] for k in ("run_id", "task", "condition", "model", "rep")},
                                                   "at": now(), "file": False, "reason": f"no submission after {attempts} attempts"})


def run_batch(cfg: dict, batch_id: str, only_models: list[str] | None = None) -> list[dict]:
    """Runs every planned run without a submission. Runs that ended in an error are retried
    on the next call, within the submission limit; a run out of attempts gets an empty submission."""
    run_dir, truth_dir = batch_dirs(cfg, batch_id)
    manifest = read_json(run_dir / "manifest.json")
    run_baselines(cfg, batch_id)
    models = {m["name"]: m for m in cfg["models"]}
    submitted = {s["run_id"] for s in read_jsonl(truth_dir / "submissions.jsonl")}
    records = []
    for run in manifest["runs"]:
        if run["run_id"] in submitted or (only_models and run["model"] not in only_models):
            continue
        spec, mcfg = manifest["specs"][run["task"]], models[run["model"]]
        attempt = 1 + sum(r["run_id"] == run["run_id"] for r in read_jsonl(run_dir / "runs.jsonl"))
        if attempt > cfg["run"]["max_attempts"]:  # only if max_attempts was lowered after the last attempt
            _out_of_attempts(truth_dir, run, attempt - 1)
            continue
        base = run_dir / run["run_id"] / f"a{attempt}"
        ws, logs = base / "workspace", base / "logs"
        isolated = mcfg.get("isolation") == "agent_user"
        with agentuser.workspace(cfg, ws, mcfg.get("keep", [])) if isolated else nullcontext(ws) as live:
            stage(spec, cfg, cfg["conditions"][run["condition"]], live, _seed(manifest["plan"].get("seed", 0), run["run_id"]),
                  expected=manifest["inputs_sha256"][run["task"]])
            found = instruction_files(live)
            if found:
                raise RunError(f"agent instruction files reachable from the workspace: {found}")
            backend = make_backend(mcfg, cfg["conditions"][run["condition"]]["mode"], cfg)
            wrap = None if isolated else sandbox.wrapper(cfg, live, network=mcfg.get("sandbox_network", False))
            res = backend.run(cfg["prompt"], live, timeout_s=spec["budgets"]["wall_time_s"], wrap=wrap)
        logs.mkdir(parents=True)
        (logs / "stdout.txt").write_text(res.stdout)
        (logs / "stderr.txt").write_text(res.stderr)
        status = "timeout" if res.timed_out else ("ok" if res.returncode == 0 else "error")
        rec = {**run, "attempt": attempt, "status": status, "returncode": res.returncode,
               "wall_s": round(res.seconds, 2), "tokens": parse_usage(backend.usage, res.stdout),
               "gpu_hours": None, "version": backend.version(), "backend": backend.describe(), "at": now()}
        if status != "error":
            sub = submit(truth_dir, run, ws, spec["budgets"]["submission_limit"],
                         extra={"attempt": attempt, "status": status})
            rec["submitted"] = sub["file"]
        append_jsonl(run_dir / "runs.jsonl", rec)
        records.append(rec)
        if status == "error" and attempt >= cfg["run"]["max_attempts"]:
            _out_of_attempts(truth_dir, run, attempt)
    return records


def run_until_done(cfg: dict, batch_id: str, only_models: list[str] | None = None, sleep=time.sleep):
    """Calls run_batch every run.retry_interval_s seconds until no planned run (of these models) is missing.
    Ends: each pass submits a run or uses one of its attempts, and a run out of attempts gets an empty submission."""
    run_dir, truth_dir = batch_dirs(cfg, batch_id)
    runs = [r for r in read_json(run_dir / "manifest.json")["runs"] if not only_models or r["model"] in only_models]
    while True:
        yield from run_batch(cfg, batch_id, only_models)
        done = {s["run_id"] for s in read_jsonl(truth_dir / "submissions.jsonl")}
        if all(r["run_id"] in done for r in runs):
            return
        sleep(cfg["run"]["retry_interval_s"])
