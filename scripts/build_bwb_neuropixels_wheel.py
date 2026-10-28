"""Build bwb_neuropixels_wheel under data_root and truth_root from the benchmark's processed
evaluation sessions (all_units build, s3://brain-wide-bench; docs/DATA.md). Needs h5py."""
import argparse
from pathlib import Path

from nabench.config import load_config
from nabench.tasks.bwb_neuropixels_wheel import build

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", default=None)
ap.add_argument("--raw", default=None,
                help="folder of the 29 eval .h5 files (default data_root/raw/bwb/all_units/ibl_brain_wide_bench_2026/eval)")
args = ap.parse_args()
cfg = load_config(args.config)
data, truth = (Path(cfg["paths"][k]) / "tasks" / "bwb_neuropixels_wheel" for k in ("data_root", "truth_root"))
raw = Path(args.raw or Path(cfg["paths"]["data_root"]) / "raw" / "bwb" / "all_units" / "ibl_brain_wide_bench_2026" / "eval")
build(raw, data, truth)
print(f"bwb_neuropixels_wheel written to {data} and {truth}")
