"""The bwb_neuropixels_wheel SOTA rerun (POYO+, 5 finetuning seeds) on a CUDA machine (D-010).

`scripts/sota_gpu.py export` copies this file into the export as run_sota.py. On the GPU machine (Linux, Python 3.12):

    python3.12 -m venv env && env/bin/pip install -r EXPORT/requirements.txt   # PyPI's Linux torch wheels include CUDA
    env/bin/python -I EXPORT/run_sota.py EXPORT --smoke   # 3 epochs of one session: checks CUDA, projects the run time
    env/bin/python -I EXPORT/run_sota.py EXPORT OUT       # the full run; rerun the same command to resume

The full run checks EXPORT/MANIFEST.sha256, finetunes POYO+ per session at the 5 eval seeds (each finished fit is kept
in OUT/parts/) and writes OUT/predictions.csv (the 5-seed ensemble), OUT/seeds/seed<k>.csv, OUT/provenance.json and
OUT/MANIFEST.sha256. Bring OUT back without parts/ and run `scripts/sota_gpu.py import OUT` on the harness. No labels
are in the export; scoring happens on the harness.
"""
import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("export")
ap.add_argument("out", nargs="?")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if not args.smoke and not args.out:
    ap.error("OUT is required without --smoke")
exp = Path(args.export).resolve()
listed = {rel: h for h, rel in (line.split("  ", 1) for line in (exp / "MANIFEST.sha256").read_text().splitlines() if line)}
bad = [rel for rel, h in listed.items() if not (exp / rel).is_file() or sha256(exp / rel) != h]
if bad:
    sys.exit(f"{len(bad)} export file(s) missing or not matching MANIFEST.sha256, e.g. {bad[:3]}")

sys.path.insert(0, str(exp / "code"))
import torch  # noqa: E402

from nabench.tasks import bwb_neuropixels_wheel as bwb  # noqa: E402

if not torch.cuda.is_available():
    sys.exit("no CUDA device")
root, src = exp / "tasks" / "bwb_neuropixels_wheel", exp / "src" / "ibl-bwb" / "src"
cases = bwb.load_cases(exp / "workspace")

if args.smoke:
    c = cases[0]
    tr, va = ~c["is_val"], c["is_val"]
    stamps = []
    bwb.poyo_finetune(bwb.load_checkpoint(root / "tools" / "poyo_plus" / "checkpoint.pt"), src,
                      bwb._pick(c["dev"], tr), c["y"][tr], bwb._pick(c["dev"], va), c["y"][va], c["test"][:1],
                      c["n_units"], base_lr=bwb._published(root, "poyo_plus_lrs.json", c["id"])["base_lr"],
                      seed=bwb.EVAL_SEEDS[0], device=torch.device("cuda"), name=c["id"], num_epochs=3,
                      epoch_log=lambda epoch, score: stamps.append(time.time()))
    per_trial = (stamps[2] - stamps[1]) / tr.sum()  # epochs 2-3, after warm-up
    total = per_trial * sum(int((~x["is_val"]).sum()) for x in cases) * 93 * bwb.SOTA_SEEDS  # published runs: 93 epochs on average
    print(f"{torch.cuda.get_device_name(0)}: {per_trial * tr.sum():.1f} s per epoch on a session of {tr.sum()} train "
          f"trials; projected full run about {total / 3600:.1f} h (from frozen-backbone epochs; from epoch 40 everything "
          f"trains, so the run takes longer)")
    sys.exit(0)

out = Path(args.out).resolve()
(out / "parts").mkdir(parents=True, exist_ok=True)
started, t0, resumed = dt.datetime.now(dt.timezone.utc), time.time(), len(list((out / "parts").glob("*.npy")))
preds = bwb._sota(cases, out, {}, root, cache=out / "parts")
bwb.write_cases(out / "predictions.csv", {cid: bwb._fmt(p) for cid, p in preds.items()})
(out / "provenance.json").write_text(json.dumps({
    "inputs_sha256": {rel: listed[f"workspace/data/{rel}"] for rel in bwb.SOTA_INPUTS},
    "export_manifest_sha256": sha256(exp / "MANIFEST.sha256"),
    "seeds": list(bwb.EVAL_SEEDS[:bwb.SOTA_SEEDS]), "selection": bwb.SELECTION, "precision": "fp32",
    "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
    "python": platform.python_version(), "started": started.isoformat(timespec="seconds"),
    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    "wall_s": round(time.time() - t0), "fits_resumed_from_parts": resumed}, indent=1))
bwb.write_manifest(out, bwb.SOTA_RESULT)
print(f"wrote {out}; bring it back without parts/ and run scripts/sota_gpu.py import")
