"""Run every planned run of a batch that has no submission yet (baselines first)."""
import argparse

from nabench.config import load_config
from nabench.runner import run_batch

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", default=None)
ap.add_argument("--batch", required=True)
ap.add_argument("--models", nargs="+", default=None)
args = ap.parse_args()
for rec in run_batch(load_config(args.config), args.batch, args.models):
    print(f"{rec['run_id']} attempt {rec['attempt']}: {rec['status']} in {rec['wall_s']}s")
