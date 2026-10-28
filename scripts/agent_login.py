"""Sign a model's CLI in as the agent user (models with `isolation: agent_user`).

Runs the model's `login_command` as the agent user in a throwaway workspace and home, as a run
would, and keeps only the files the model lists in `keep` (in sandbox.agent_keep) for later runs.
Run it in a terminal: the CLI prints a URL to open and may ask you to paste a code.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from nabench import agentuser
from nabench.batch import new_batch_id
from nabench.config import load_config

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", default=None)
ap.add_argument("--model", required=True)
args = ap.parse_args()
cfg = load_config(args.config)
mcfg = next((m for m in cfg["models"] if m["name"] == args.model), None)
if mcfg is None or mcfg.get("isolation") != "agent_user" or "login_command" not in mcfg:
    sys.exit(f"{args.model}: not an agent-user model with a login_command")

keep = mcfg.get("keep", [])
dest = Path(cfg["paths"]["runs_root"]) / "logins" / new_batch_id(args.model) / "workspace"
with agentuser.workspace(cfg, dest, keep) as ws:
    ws.mkdir()
    rc = subprocess.run([*agentuser.prefix(cfg, mcfg.get("env")), *mcfg["login_command"]], cwd=ws).returncode
store = Path(cfg["sandbox"]["agent_keep"])
kept = [rel for rel in keep if (store / rel).is_file()]
print(f"{args.model}: login exited {rc}; kept for later runs: {', '.join(kept) or 'nothing'}")
sys.exit(1 if rc or not kept else 0)
