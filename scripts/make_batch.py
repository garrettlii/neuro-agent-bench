"""Create a batch from a pre-registration (plan and predictions in its yaml block),
or a smoke batch on synthetic tasks only."""
import argparse
import sys

from nabench.batch import LedgerError
from nabench.config import load_config
from nabench.prereg import PreregError, check_prereg
from nabench.runner import RunError, create_batch
from nabench.spec import SpecError

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", default=None)
ap.add_argument("--prereg", help="pre-registration file, committed and pushed")
ap.add_argument("--smoke", action="store_true", help="synthetic tasks only; no pre-registration")
ap.add_argument("--tasks", nargs="+", default=["dummy_spikes"])
ap.add_argument("--conditions", nargs="+", default=["C1", "C2", "C3"])
ap.add_argument("--models", nargs="+", default=["fake-scripted"])
ap.add_argument("--additional-batch-reason", default=None,
                help="required to generate a second batch under the same pre-registration commit")
args = ap.parse_args()
if bool(args.prereg) == args.smoke:
    sys.exit("pass exactly one of --prereg or --smoke")
cfg = load_config(args.config)
try:
    if args.smoke:
        prereg = None
        plan = {"seed": 0, "blocks": [{"tasks": args.tasks, "conditions": args.conditions, "models": args.models}]}
    else:
        prereg = check_prereg(args.prereg)
        plan = prereg["plan"]
        print(f"pre-registration OK: {prereg['path']} @ {prereg['commit'][:10]} ({prereg['committed_at']})")
    batch_id = create_batch(cfg, plan, prereg, args.additional_batch_reason)
except (PreregError, LedgerError, RunError, SpecError) as exc:
    sys.exit(f"refused: {exc}")
print(f"batch {batch_id}")
