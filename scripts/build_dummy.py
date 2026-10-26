"""Generate the synthetic dummy_spikes task under data_root and truth_root."""
import argparse
from pathlib import Path

from nabench.config import load_config
from nabench.tasks.dummy_spikes import build

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
data, truth = (Path(cfg["paths"][k]) / "tasks" / "dummy_spikes" for k in ("data_root", "truth_root"))
build(data, truth)
print(f"dummy_spikes written to {data} and {truth}")
