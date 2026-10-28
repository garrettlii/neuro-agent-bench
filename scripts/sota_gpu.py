"""bwb_neuropixels_wheel SOTA on a separate GPU machine (D-010): export its inputs, import its predictions.

    python scripts/sota_gpu.py export DIR --dry-run   # list what would be copied, with sizes; writes nothing
    python scripts/sota_gpu.py export DIR             # write the export to DIR, with MANIFEST.sha256
    python scripts/sota_gpu.py import OUT             # check the job's output and place it for the baseline step

The export holds the dev and test inputs the job reads (never truth), the POYO+ checkpoint, the selected learning
rates, the ibl-bwb source at the spec's commit, this package's BWB task module, the job script (scripts/sota_gpu_job.py)
and its environment. Import checks OUT's manifest, that the job read exactly the local inputs, and the predictions'
format, then copies them to data_root/tasks/bwb_neuropixels_wheel/sota_gpu/, where sota_baseline reads them.
"""
import argparse
import io
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from nabench.config import REPO_ROOT, load_config
from nabench.spec import load_spec, resolve
from nabench.tasks import bwb_neuropixels_wheel as bwb

CODE = ("__init__.py", "metrics.py", "tasks/__init__.py", "tasks/bwb_neuropixels_wheel.py")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("action", choices=["export", "import"])
ap.add_argument("dir")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
data_root = Path(cfg["paths"]["data_root"])
root = data_root / "tasks" / "bwb_neuropixels_wheel"
target = Path(args.dir).resolve()

if args.action == "import":
    errors = bwb.check_sota_result(target, root)
    dest = root / "sota_gpu"
    if errors:
        sys.exit("not imported:\n- " + "\n- ".join(errors))
    if dest.exists():
        sys.exit(f"{dest} exists; move it aside first")
    for f in (*bwb.SOTA_RESULT, "MANIFEST.sha256"):
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target / f, dest / f)
    errors = bwb.check_sota_result(dest, root)
    if errors:
        sys.exit(f"copy in {dest} failed its check:\n- " + "\n- ".join(errors))
    print(f"imported to {dest}: {len(bwb.SOTA_RESULT) - 1} prediction files, checked against the local inputs")
    sys.exit(0)

spec = resolve(load_spec("bwb_neuropixels_wheel"), cfg)
commit = re.search(r"@ ([0-9a-f]{40})", spec["baselines"]["sota"]["code"]).group(1)
ibl = data_root / "src" / "ibl-bwb"
staged = {Path(p).parent.name + "/" + Path(p).name: Path(p)
          for k in ("dev_inputs", "dev_labels", "test_inputs") for p in spec["data"][k]}
files = {f"workspace/data/{rel}": staged[rel] for rel in bwb.SOTA_INPUTS}
files |= {f"tasks/bwb_neuropixels_wheel/{rel}": root / rel
          for rel in ("tools/poyo_plus/checkpoint.pt", "reference/poyo_plus_lrs.json")}
files |= {f"code/nabench/{rel}": REPO_ROOT / "src" / "nabench" / rel for rel in CODE}
files["run_sota.py"] = REPO_ROOT / "scripts" / "sota_gpu_job.py"
truth_root = Path(cfg["paths"]["truth_root"]).resolve()
assert not any(p.resolve().is_relative_to(truth_root) for p in files.values()), "truth must never be exported"
freeze = subprocess.run(["uv", "pip", "freeze", "--python", str(data_root / "envs" / "bwb" / "bin" / "python")],
                        capture_output=True, text=True, check=True).stdout
requirements = "".join(line + "\n" for line in freeze.splitlines() if line and not line.startswith("-e "))
archive = subprocess.run(["git", "-C", str(ibl), "archive", commit, "src"], capture_output=True, check=True).stdout
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    members = [m for m in tar.getmembers() if m.isfile()]

sizes = {rel: src.stat().st_size for rel, src in files.items()}
print(f"export of {len(files) + len(members) + 1} files to {target}:")
for rel, n in sizes.items():
    print(f"  {n / 1e6:10.1f} MB  {rel}")
print(f"  {sum(m.size for m in members) / 1e6:10.1f} MB  src/ibl-bwb/src/ ({len(members)} files, ibl-bwb @ {commit[:10]})")
print(f"  {len(requirements) / 1e6:10.1f} MB  requirements.txt ({len(requirements.splitlines())} pins, the bwb environment)")
print(f"  total {(sum(sizes.values()) + sum(m.size for m in members) + len(requirements)) / 1e9:.2f} GB, plus MANIFEST.sha256")
if args.dry_run:
    sys.exit(0)

if target.exists():
    sys.exit(f"{target} exists; choose a new folder")
target.mkdir(parents=True)
for rel, src in files.items():
    (target / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, target / rel)
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    tar.extractall(target / "src" / "ibl-bwb", members=members, filter="data")
(target / "requirements.txt").write_text(requirements)
bwb.write_manifest(target, [*files, *(f"src/ibl-bwb/{m.name}" for m in members), "requirements.txt"])
print(f"wrote {target}; nothing has been transferred")
