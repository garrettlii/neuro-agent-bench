"""Scorer. Runs outside the sandbox: reads truth_root, writes per-case scores
privately under truth_root/<batch>/scores/ and aggregates only to reports/.

Normalized score = (agent - simple) / (SOTA - simple): 0 = simple baseline,
1 = matches the SOTA rerun, > 1 beats it. Also against the noise ceiling where
one exists. Metrics are per-case and aggregate by the mean, so every CI is a
paired bootstrap over test cases (the same resampled cases for the agent and
both baselines). The report is final only when nothing is missing: every
planned run and baseline has a scored submission.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .batch import batch_dirs, read_json, read_jsonl
from .config import REPO_ROOT
from .metrics import read_cases, read_predictions, write_cases
from .runner import BASELINES, run_id
from .spec import ref

REPORTS = REPO_ROOT / "reports"
GAPS = (("C1", "C2"), ("C1", "C3"))


def ratio(a: np.ndarray, s: np.ndarray, o: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - s) / (o - s)


def ci(draws: np.ndarray, point: float) -> tuple[float, float, float]:
    draws = draws[np.isfinite(draws)]
    if not len(draws) or not np.isfinite(point):
        return point, np.nan, np.nan
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def fmt(t) -> str:
    p, lo, hi = t
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else f"{p:.3f}"


CACHED_INPUT = ("cache_creation_input_tokens", "cache_read_input_tokens")  # Claude counts these apart from input_tokens;
# Codex's input_tokens already includes its cached_input_tokens


def _tokens(records) -> str:
    us = [r["tokens"] for r in records if r.get("tokens")]
    if not us:
        return "n/a"
    tokens_in = sum(u.get("input_tokens", 0) + sum(u.get(k, 0) for k in CACHED_INPUT) for u in us)
    return f"{tokens_in:,.0f} / {sum(u.get('output_tokens', 0) for u in us):,.0f}"


def completeness(manifest: dict, submissions: list[dict], runs: list[dict]) -> dict:
    planned = [r["run_id"] for r in manifest["runs"]]
    planned += [run_id(t, "baseline", k, 1) for t in manifest["specs"] for k in BASELINES]
    planned += [run_id(t, "reference", r["name"], 1) for t, s in manifest["specs"].items() for r in s.get("references", [])]
    done = {s["run_id"] for s in submissions}
    errors = {r["run_id"] for r in runs if r["status"] == "error"} - done
    return {"planned": len(planned), "submitted": len(done & set(planned)),
            "missing": sorted(set(planned) - done), "errors_pending": len(errors)}


def score_batch(cfg: dict, batch_id: str, out_dir: Path = REPORTS, n_boot: int = 2000, seed: int = 0) -> tuple[Path, bool]:
    run_dir, truth_dir = batch_dirs(cfg, batch_id)
    manifest = read_json(truth_dir / "manifest.json")
    subs = read_jsonl(truth_dir / "submissions.jsonl")
    runs = read_jsonl(run_dir / "runs.jsonl")
    status = completeness(manifest, subs, runs)
    final = not status["missing"]
    rng = np.random.default_rng(seed)

    sections, norm_draws = [], {}
    for task, spec in manifest["specs"].items():
        truth = read_cases(spec["data"]["truth"], "label")
        metric = ref(spec["metric"]["function"])
        per_case, n_pred = {}, {}
        for s in (s for s in subs if s["task"] == task):
            path = truth_dir / "submissions" / f"{s['run_id']}.csv"
            preds, _ = read_predictions(path if s["file"] else None, truth)
            n_pred[s["run_id"]] = len(preds)
            per_case[s["run_id"]] = metric(preds, truth)
            write_cases(truth_dir / "scores" / f"{s['run_id']}.csv", dict(zip(truth, per_case[s["run_id"]])), "score")
        sections += [f"## {task} ({spec['modality']}, selection: {spec['selection']})", ""]
        base = {k: per_case.get(run_id(task, "baseline", k, 1)) for k in BASELINES}
        if base["simple"] is None or base["sota"] is None:
            sections += ["Baselines not yet scored; no normalized scores.", ""]
            continue
        idx = rng.integers(0, len(truth), (n_boot, len(truth)))
        S, O = base["simple"], base["sota"]
        ceil_fn = spec["noise_ceiling"]["function"]
        C = metric(ref(ceil_fn)(spec), truth) if ceil_fn else None

        def norm(a, ref_scores):
            return ci(ratio(a[idx].mean(1), S[idx].mean(1), ref_scores[idx].mean(1)),
                      float(ratio(a.mean(), S.mean(), ref_scores.mean())))

        pub = spec["baselines"]["sota"]["published"]
        P = pub["value"] if isinstance(pub["value"], (int, float)) else None

        def vs_pub(a):  # the published SOTA value in place of the rerun (D-007); the rerun stays primary
            if P is None:
                return "n/a"
            return fmt(ci(ratio(a[idx].mean(1), S[idx].mean(1), P), float(ratio(a.mean(), S.mean(), P))))

        sections += [f"Metric: {spec['metric']['description']} ({len(truth)} test cases). "
                     f"SOTA rerun ({spec['baselines']['sota']['name']}): {O.mean():.3f}; published: "
                     f"{pub['value'] if pub['value'] is not None else 'none'} ({pub['source']})."
                     + (f" Hardware: {spec['baselines']['sota']['hardware']}" if "hardware" in spec["baselines"]["sota"] else ""), "",
                     "| run | condition | model | reps | raw | normalized vs SOTA | normalized vs published SOTA | normalized vs ceiling | missing cases | tokens in / out | wall s (mean) | GPU-h |",
                     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in BASELINES:
            a = base[k]
            if a is not None:
                sections.append(f"| baseline | - | {k} | 1 | {a.mean():.3f} | {fmt(norm(a, O))} | {vs_pub(a)} | "
                                f"{fmt(norm(a, C)) if C is not None else 'n/a'} | - | - | - | - |")
        refs = {r["name"]: r for r in spec.get("references", [])}
        ref_scores = {n: per_case[run_id(task, "reference", n, 1)] for n in refs if run_id(task, "reference", n, 1) in per_case}
        for n, a in ref_scores.items():
            sections.append(f"| reference | - | {n} | 1 | {a.mean():.3f} | {fmt(norm(a, O))} | {vs_pub(a)} | "
                            f"{fmt(norm(a, C)) if C is not None else 'n/a'} | - | - | - | - |")
        if C is not None:
            sections.append(f"| ceiling | - | {spec['noise_ceiling']['method']} | - | {C.mean():.3f} | {fmt(norm(C, O))} | {vs_pub(C)} | 1.000 | - | - | - | - |")
        groups = {}
        for r in (r for r in manifest["runs"] if r["task"] == task and r["run_id"] in per_case):
            groups.setdefault((r["condition"], r["model"]), []).append(r["run_id"])
        means = {}
        for (cond, model), rids in sorted(groups.items()):
            a = np.mean([per_case[i] for i in rids], axis=0)  # reps averaged per case
            means[(cond, model)] = a
            recs = [r for r in runs if r["run_id"] in rids]
            n_missing = sum(len(truth) - n_pred[i] for i in rids)
            walls = [r["wall_s"] for r in recs]
            gpu = [r["gpu_hours"] for r in recs if r.get("gpu_hours") is not None]
            draws = ratio(a[idx].mean(1), S[idx].mean(1), O[idx].mean(1))
            norm_draws.setdefault((spec["selection"], cond, model), {})[task] = (draws, float(ratio(a.mean(), S.mean(), O.mean())))
            sections.append(f"| agent | {cond} | {model} | {len(rids)} | {a.mean():.3f} | {fmt(norm(a, O))} | {vs_pub(a)} | "
                            f"{fmt(norm(a, C)) if C is not None else 'n/a'} | {n_missing} | {_tokens(recs)} | "
                            f"{np.mean(walls):.1f} | {sum(gpu) if gpu else 'n/a'} |")
        seeds = [a for n, a in ref_scores.items() if n.startswith("sota_seed")]
        if seeds:
            per_seed = np.mean(seeds, axis=0)
            sections += ["", f"Mean of the {len(seeds)} per-seed SOTA scores (raw, over cases and seeds): "
                             f"{fmt(ci(per_seed[idx].mean(1), float(per_seed.mean())))}"]

        def gap(a, b):  # normalized a - b, paired over cases
            return fmt(ci(ratio(a[idx].mean(1), S[idx].mean(1), O[idx].mean(1)) - ratio(b[idx].mean(1), S[idx].mean(1), O[idx].mean(1)),
                          float(ratio(a.mean(), S.mean(), O.mean()) - ratio(b.mean(), S.mean(), O.mean()))))

        models = list(dict.fromkeys(m for b in manifest["plan"]["blocks"] for m in b["models"]))  # plan order
        gap_lines = [f"- {m}: {x} - {y} = {gap(means[(x, m)], means[(y, m)])}"
                     for x, y in GAPS for m in models if (x, m) in means and (y, m) in means]
        if base["automl"] is not None:
            gap_lines += [f"- {m}: C2 - automl = {gap(means[('C2', m)], base['automl'])}" for m in models if ("C2", m) in means]
        for cond in ("C1", "C2"):
            have = [m for m in models if (cond, m) in means]
            gap_lines += [f"- {cond}: {a} - {b} = {gap(means[(cond, a)], means[(cond, b)])}" for i, a in enumerate(have) for b in have[i + 1:]]
        sections += ["", "Gaps (normalized vs SOTA, paired over cases):" if gap_lines else "", *gap_lines, ""]

        signal_free = next((n for n, r in refs.items() if "leakage_margin" in r and n in ref_scores), None)
        leak_lines = []
        for m in (m for m in models if ("C3", m) in means):
            p, lo, _ = norm(means[("C3", m)], O)
            line, hit = f"- {m}: C3 lower CI bound {lo:.3f}, {'above' if lo > 0 else 'not above'} 0", lo > 0
            if signal_free:
                margin, d = refs[signal_free]["leakage_margin"], p - norm(ref_scores[signal_free], O)[0]
                line += f"; C3 - {signal_free} = {d:.3f}, {'more' if d > margin else 'not more'} than {margin}"
                hit = hit or d > margin
            leak_lines.append(f"{line}: {'TRIGGERED' if hit else 'not triggered'}")
        if leak_lines:
            sections += ["Leakage check (C3 normalized vs SOTA):", *leak_lines, ""]

    agg = ["## Aggregate across tasks", "",
           "Mean normalized score (vs SOTA) over tasks, bootstrap resampling cases within each task. "
           "Groups with different selection are never pooled.", "",
           "| selection | condition | model | tasks | mean normalized |", "|---|---|---|---|---|"]
    for (sel, cond, model), per_task in sorted(norm_draws.items()):
        draws = np.mean([d for d, _ in per_task.values()], axis=0)
        agg.append(f"| {sel} | {cond} | {model} | {len(per_task)} | {fmt(ci(draws, float(np.mean([p for _, p in per_task.values()]))))} |")

    kind = "Smoke batch (synthetic tasks, no pre-registration)" if manifest["smoke"] else \
        f"Pre-registration {manifest['prereg']['path']} @ {manifest['prereg']['commit'][:10]}"
    head = [f"# Results: batch {batch_id}", "", f"{kind}. Aggregates only; per-case scores stay under truth_root.", ""]
    if final:
        head += [f"Completeness: {status['submitted']}/{status['planned']} runs and baselines scored.", ""]
    else:
        head += [f"> **INTERIM.** {len(status['missing'])} of {status['planned']} planned runs and baselines have no scored "
                 f"submission ({status['errors_pending']} with errors pending retry). Only the final report counts.", ""]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{batch_id}.md"
    out.write_text("\n".join(head + sections + agg) + "\n")
    return out, final
