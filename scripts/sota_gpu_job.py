"""The bwb_neuropixels_wheel SOTA rerun (POYO+, 5 seeds) on CUDA GPUs (D-010, D-018, D-019).

`scripts/sota_gpu.py export` copies this file into the export as run_sota.py. On the GPU machine(s) (Linux, Python 3.12):

    python3.12 -m venv env && env/bin/pip install -r EXPORT/requirements.txt   # PyPI's Linux torch wheels include CUDA
    env/bin/python -I EXPORT/run_sota.py EXPORT --smoke             # projected seconds per session per seed
    CUDA_VISIBLE_DEVICES=0 env/bin/python -I EXPORT/run_sota.py EXPORT PART1 --seeds 1 2 3
    CUDA_VISIBLE_DEVICES=1 env/bin/python -I EXPORT/run_sota.py EXPORT PART2 --seeds 4 5
    env/bin/python -I EXPORT/run_sota.py EXPORT OUT --merge PART1 PART2

Seeds may run in parallel, on separate GPUs or as concurrent processes on one; without --seeds a process runs all
five, one after another. The 4 h limit covers the whole job, from the first seed's start to the last seed's end
(D-019); across machines that relies on their clocks agreeing. Each process stops at the first epoch end more than
4 h after its own start. If a seed does not finish every session by then, the process writes nothing and exits with
status 3: the job stops, nothing is imported, and if it was the process's first seed, BWB fails rule (c) on that
hardware. There is no partial-seed result.

A run checks EXPORT/MANIFEST.sha256 and finetunes POYO+ on every session for each of its seeds, in bf16 autocast
as the benchmark did when the GPU supports bf16, else in fp32 (D-018). It writes PART/seeds/seed<k>.csv,
PART/part.json (each seed's GPU, precision, start and end) and PART/MANIFEST.sha256. --merge checks the parts and
writes the standard result: OUT/predictions.csv (the mean of the 5 seeds), OUT/seeds/seed1..5.csv,
OUT/provenance.json (each seed's GPU, precision, start and end, and the job's overall wall time) and
OUT/MANIFEST.sha256. It refuses parts from another export, missing or repeated seeds, and a job over 4 h. Bring OUT
back and run `scripts/sota_gpu.py import OUT` on the harness. No labels are in the export; scoring happens on the
harness.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from pathlib import Path

START = time.time()  # this process's 4 h deadline counts from here


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def checked(folder: Path) -> dict[str, str]:
    """folder's MANIFEST.sha256 as {path: hash}, after checking every listed file against it."""
    listed = {rel: h for h, rel in (line.split("  ", 1) for line in (folder / "MANIFEST.sha256").read_text().splitlines() if line)}
    bad = [rel for rel, h in listed.items() if not (folder / rel).is_file() or sha256(folder / rel) != h]
    if bad:
        sys.exit(f"{folder}: {len(bad)} file(s) missing or not matching MANIFEST.sha256, e.g. {bad[:3]}")
    return listed


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def stop(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(3)


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("export")
ap.add_argument("out", nargs="?", help="PART for a run, OUT for --merge")
ap.add_argument("--seeds", nargs="+", type=int, choices=range(1, 6), metavar="S", help="seeds to run (1-5; default all)")
ap.add_argument("--merge", nargs="+", metavar="PART", help="merge these parts into OUT")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if not args.smoke and not args.out:
    ap.error("PART or OUT is required without --smoke")
exp = Path(args.export).resolve()
out = Path(args.out).resolve() if args.out else None
if out and out.exists():
    sys.exit(f"{out} exists; choose a new folder")
sys.path.insert(0, str(exp / "code"))

if args.merge:
    from nabench.metrics import read_cases
    from nabench.tasks import bwb_neuropixels_wheel as bwb
    from nabench.tasks import bwb_sota_gpu as rule

    folders = [Path(p).resolve() for p in args.merge]
    for d in folders:
        if "part.json" not in checked(d):
            sys.exit(f"{d}: no part.json in its manifest")
    parts = [json.loads((d / "part.json").read_text()) for d in folders]
    problems = rule.part_problems(parts, sha256(exp / "MANIFEST.sha256"))
    prov = None if problems else rule.merge_provenance(parts)
    problems = problems or rule.provenance_problems(prov, bwb.EVAL_SEEDS)
    if problems:
        sys.exit("not merged:\n- " + "\n- ".join(problems))
    (out / "seeds").mkdir(parents=True)
    for d, p in zip(folders, parts):
        for k in p["seeds"]:
            shutil.copyfile(d / "seeds" / f"seed{k}.csv", out / "seeds" / f"seed{k}.csv")
    seeds = [{c: bwb._trace(v) for c, v in read_cases(out / "seeds" / f"seed{k}.csv", "prediction").items()}
             for k in rule.ALL_SEEDS]
    bwb.write_cases(out / "predictions.csv", {c: bwb._fmt(sum(s[c] for s in seeds) / len(seeds)) for c in seeds[0]})
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    bwb.write_manifest(out, bwb.SOTA_RESULT)
    print(f"wrote {out}: 5 seeds, job wall time {prov['wall_s']:.0f} s of {rule.LIMIT_S}; run scripts/sota_gpu.py import")
    sys.exit(0)

listed = checked(exp)
import torch  # noqa: E402

from nabench.tasks import bwb_neuropixels_wheel as bwb  # noqa: E402
from nabench.tasks import bwb_sota_gpu as rule  # noqa: E402

if not torch.cuda.is_available():
    sys.exit("no CUDA device")
assert bwb.SELECTION == "published"  # D-012: the benchmark's selected learning rates
device, gpu = torch.device("cuda"), torch.cuda.get_device_name(0)
bf16 = bool(torch.cuda.is_bf16_supported())
if bf16:
    # bf16-mixed as the benchmark trained: the forward pass under autocast, fp32 weights, outputs and loss. Patched
    # here, not in the task module, which the running batch imports unchanged.
    _load = bwb._ibl_bwb

    def _ibl_bwb_bf16(src):
        POYOPlus, spec, groups = _load(src)

        class POYOPlusBF16(POYOPlus):
            def forward(self, *a, **kw):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return super().forward(*a, **kw).float()

        return POYOPlusBF16, spec, groups

    bwb._ibl_bwb = _ibl_bwb_bf16
precision = "bf16" if bf16 else "fp32"
root, src = exp / "tasks" / "bwb_neuropixels_wheel", exp / "src" / "ibl-bwb" / "src"
ckpt = bwb.load_checkpoint(root / "tools" / "poyo_plus" / "checkpoint.pt")
cases = {c["id"]: c for c in bwb.load_cases(exp / "workspace")}


def fit(cid: str, k: int, deadline: float, targets=None, **kw):
    """POYO+ finetuned on one case at seed k, as _sota does it; raises rule.TimeUp at the first epoch end past
    the deadline."""
    if time.time() > deadline:
        raise rule.TimeUp

    def tick(epoch, score):
        if time.time() > deadline:
            raise rule.TimeUp

    c = cases[cid]
    tr, va = ~c["is_val"], c["is_val"]
    return bwb.poyo_finetune(ckpt, src, bwb._pick(c["dev"], tr), c["y"][tr], bwb._pick(c["dev"], va), c["y"][va],
                             c["test"] if targets is None else targets, c["n_units"],
                             base_lr=bwb._published(root, "poyo_plus_lrs.json", cid)["base_lr"],
                             seed=bwb.EVAL_SEEDS[k - 1], device=device, name=cid, epoch_log=kw.pop("epoch_log", tick), **kw)[1]


if args.smoke:
    # 42 epochs of one session: epochs 2-39 train only the embeddings, epochs 40-41 everything (poyo_finetune)
    cid = next(iter(cases))
    stamps = []
    fit(cid, 1, float("inf"), targets=cases[cid]["test"][:1], num_epochs=42, epoch_log=lambda e, s: stamps.append(time.time()))
    frozen, full = (stamps[39] - stamps[1]) / 38, (stamps[41] - stamps[39]) / 2
    n = int((~cases[cid]["is_val"]).sum())
    # published runs: 93 epochs per session and seed on average; epoch time taken as proportional to train trials
    per_session = sorted((40 * frozen + 53 * full) * int((~c["is_val"]).sum()) / n for c in cases.values())
    seed_s = sum(per_session)
    print(f"{gpu}, {precision}: {frozen:.2f} s per epoch with the backbone frozen, {full:.2f} s with everything "
          f"training, on a session of {n} train trials")
    print(f"projected seconds per session per seed (93 epochs, scaled by train trials): mean {seed_s / len(cases):.0f}, "
          f"min {per_session[0]:.0f}, max {per_session[-1]:.0f}; one seed over all {len(cases)} sessions: {seed_s:.0f} s")
    fit_one = int(rule.LIMIT_S // seed_s)
    print(f"seeds one GPU like this fits in {rule.LIMIT_S} s, one after another: {fit_one}; " +
          (f"GPUs for 5 seeds in parallel: {math.ceil(5 / fit_one)}" if fit_one else
           "a single seed is projected over the limit: under D-019 BWB would fail rule (c) on this GPU") +
          " (concurrent processes on one GPU each run slower; run the smoke concurrently to measure that)")
    sys.exit(0)

seeds = sorted(set(args.seeds or rule.ALL_SEEDS))
deadline, results, record = START + rule.LIMIT_S, {}, {}
for i, k in enumerate(seeds):
    t, started = time.time(), now()
    try:
        results[k] = {cid: fit(cid, k, deadline) for cid in cases}
    except rule.TimeUp:
        stop(f"seed {k} did not finish all {len(cases)} sessions within {rule.LIMIT_S} s of this process's start on "
             f"{gpu} ({precision}); nothing written. Under D-019 the job stops and nothing is imported" +
             ("; a single seed did not fit, so BWB fails rule (c) on this hardware." if i == 0 else
              f". It was seed {i + 1} of {len(seeds)} in this process; with more GPUs, run fewer seeds per process."))
    record[str(k)] = {"eval_seed": bwb.EVAL_SEEDS[k - 1], "device": gpu, "bf16_supported": bf16, "precision": precision,
                      "start_unix": t, "end_unix": time.time(), "started": started, "finished": now()}
(out / "seeds").mkdir(parents=True)
for k, preds in results.items():
    bwb.write_cases(out / "seeds" / f"seed{k}.csv", {cid: bwb._fmt(p) for cid, p in preds.items()})
(out / "part.json").write_text(json.dumps({
    "inputs_sha256": {rel: listed[f"workspace/data/{rel}"] for rel in bwb.SOTA_INPUTS},
    "export_manifest_sha256": sha256(exp / "MANIFEST.sha256"), "seeds": record,
    "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
    "selection": bwb.SELECTION}, indent=1))
bwb.write_manifest(out, ["part.json", *(f"seeds/seed{k}.csv" for k in results)])
print(f"wrote {out}: seed(s) {seeds} on {gpu} ({precision}); merge all parts with --merge")
