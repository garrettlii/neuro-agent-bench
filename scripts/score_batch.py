"""Score a batch: per-case scores under truth_root, aggregates in reports/<batch>.md."""
import argparse

from nabench.config import load_config
from nabench.scoring import score_batch

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", default=None)
ap.add_argument("--batch", required=True)
args = ap.parse_args()
out, final = score_batch(load_config(args.config), args.batch)
print(f"wrote {out} ({'final' if final else 'INTERIM'})")
