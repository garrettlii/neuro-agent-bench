"""Check that a model's agent CLI keeps agent shell commands offline when the OS
sandbox allows network (as it must for CLIs that reach their own API).

Stages a throwaway workspace (moved to runs_root/probes/ afterwards), runs the
model's CLI there with network allowed, isolated as in a real run (the OS sandbox,
or the separate agent user), with the standard prompt, and
records whether the provided probe script reached a public host. The record goes
to runs_root/network_probes.jsonl with the CLI's version and command. Run it once
per model and CLI version, after checking that this machine itself is online.
"""
import argparse
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

from nabench import agentuser, netprobe, sandbox
from nabench.backends import instruction_files, make_backend
from nabench.batch import append_jsonl, new_batch_id, now
from nabench.config import load_config

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", default=None)
ap.add_argument("--model", required=True)
ap.add_argument("--timeout", type=float, default=600, help="wall-time limit for the agent, in seconds")
args = ap.parse_args()
cfg = load_config(args.config)
mcfg = next((m for m in cfg["models"] if m["name"] == args.model), None)
if mcfg is None:
    sys.exit(f"unknown model {args.model}")

backend = make_backend(mcfg, cfg=cfg)
isolated = mcfg.get("isolation") == "agent_user"
base = Path(cfg["paths"]["runs_root"]) / "probes" / new_batch_id(args.model)
logs = base / "logs"
with agentuser.workspace(cfg, base / "workspace", mcfg.get("keep", [])) if isolated else nullcontext(base / "workspace") as ws:
    (ws / "submission").mkdir(parents=True)
    (ws / "probe.py").write_text(netprobe.PROBE_SCRIPT)
    # control as the same user but outside any sandbox: an offline result means nothing if that user cannot reach the host
    python = [*agentuser.prefix(cfg), "python3"] if isolated else [sys.executable, "-I"]
    subprocess.run([*python, "probe.py", netprobe.HOST, "control.json"], cwd=ws, check=True, timeout=60,
                   capture_output=True, stdin=subprocess.DEVNULL)
    control = json.loads((ws / "control.json").read_text())[0]
    (ws / "control.json").unlink()
    if (control["tcp"], control["https"]) != ("connected", "connected"):
        sys.exit(f"refused: the control run could not reach {netprobe.HOST} ({control}); nothing recorded")
    (ws / "TASK.md").write_text(netprobe.TASK)
    found = instruction_files(ws)
    if found:
        sys.exit(f"refused: agent instruction files reachable from the workspace: {found}")
    wrap = None if isolated else sandbox.wrapper(cfg, ws, network=True)
    res = backend.run(cfg["prompt"], ws, timeout_s=args.timeout, wrap=wrap)
ws = base / "workspace"
logs.mkdir()
(logs / "stdout.txt").write_text(res.stdout)
(logs / "stderr.txt").write_text(res.stderr)

out = ws / netprobe.OUTPUT
try:
    attempts = json.loads(out.read_text()) if out.is_file() and not out.is_symlink() else None
except (OSError, json.JSONDecodeError, UnicodeDecodeError):
    attempts = None
rec = {"model": args.model, "version": backend.version(), "backend": backend.describe(),
       "passed": netprobe.verdict(attempts), "at": now(), "control": control, "attempts": attempts,
       "returncode": res.returncode, "timed_out": res.timed_out, "wall_s": round(res.seconds, 2), "workspace": str(ws)}
append_jsonl(netprobe.log_path(cfg), rec)
print(f"{args.model} {rec['version']}: {'offline (passed)' if rec['passed'] else 'NOT offline or probe not run (failed)'}")
print(f"attempts: {attempts}")
