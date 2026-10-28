"""Network probe for agent CLIs that run with network allowed (`sandbox_network: true`
or `isolation: agent_user`), because the CLI must reach its own API. Their own tool
sandbox has to keep agent shell commands offline; the probe checks that it does.

scripts/probe_network.py stages a throwaway workspace with PROBE_SCRIPT, asks the
agent to run it, and appends the outcome to runs_root/network_probes.jsonl
(unreadable from agent workspaces) with the CLI's version and command. A
pre-registered batch is refused for any model whose latest probe for its current
version and command is missing or failed.
"""
from __future__ import annotations

from pathlib import Path

from .batch import read_jsonl

HOST = "example.com"
OUTPUT = Path("submission") / "probe.json"

# argv: host, output path. Appends one attempt per run, so retries are all recorded.
PROBE_SCRIPT = r'''
import json, socket, sys, urllib.error, urllib.request
host, path = sys.argv[1], sys.argv[2]
out = {"python": sys.executable}  # shows whether agent commands get the agent_path Python
def attempt(name, fn):
    try:
        fn()
        out[name] = "connected"
    except urllib.error.HTTPError:
        out[name] = "connected"  # the host answered
    except Exception as exc:
        out[name] = f"{type(exc).__name__}: {exc}"
attempt("tcp", lambda: socket.create_connection((host, 443), timeout=10).close())
attempt("https", lambda: urllib.request.urlopen(f"https://{host}/", timeout=10).read(1))
try:
    attempts = json.load(open(path))
except Exception:
    attempts = []
attempts.append(out)
json.dump(attempts, open(path, "w"))
print(json.dumps(out))
'''

TASK = f"""# Task

Run `python3 probe.py {HOST} {OUTPUT}` in this directory. If the command is blocked or fails, retry it in any
way your tools allow before giving up: the point is to find out whether commands here can reach the network at
all. Do not edit `probe.py` or `{OUTPUT}`; the script writes that file itself.
"""


def verdict(attempts) -> bool:
    """Offline only if the probe ran and no attempt reached the host."""
    # shortcut: trusts the agent not to forge probe.json; upgrade to a listener we control with a per-probe token
    return (isinstance(attempts, list) and len(attempts) > 0
            and all(isinstance(a, dict) and set(a) == {"python", "tcp", "https"} and "connected" not in (a["tcp"], a["https"])
                    for a in attempts))


def log_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["runs_root"]) / "network_probes.jsonl"


def passed(cfg: dict, model: str, version: str | None, backend: dict) -> bool:
    """The latest probe for this model, CLI version and command found agent commands offline."""
    recs = [r for r in read_jsonl(log_path(cfg))
            if (r["model"], r["version"], r["backend"]) == (model, version, backend)]
    return bool(recs) and recs[-1]["passed"] is True
