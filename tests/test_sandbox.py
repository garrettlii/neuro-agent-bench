"""The agent cannot read the scorer's files, other runs, the data store or this repo."""
import json
import sys
from pathlib import Path

import pytest

from nabench import sandbox
from nabench.backends import CLIBackend
from nabench.config import REPO_ROOT

pytestmark = pytest.mark.skipif(not sandbox.available(), reason="no supported sandbox on this platform")

PROBE = r"""
import json, os, socket, sys
targets = json.loads(sys.argv[1])
out = {}
def attempt(name, fn):
    try:
        fn(); out[name] = "allowed"
    except Exception as exc:
        out[name] = type(exc).__name__
for name, path in targets["read"].items():
    attempt("read:" + name, lambda p=path: open(p).read())
for name, path in targets["list"].items():
    attempt("list:" + name, lambda p=path: os.listdir(p))
for name, path in targets["write"].items():
    attempt("write:" + name, lambda p=path: open(p, "w").write("x"))
attempt("network", lambda: socket.create_connection(("1.1.1.1", 443), timeout=3))
print(json.dumps(out))
"""


def test_agent_cannot_read_scorer_files(cfg):
    p = {k: Path(v) for k, v in cfg["paths"].items()}
    truth = p["truth_root"] / "b1" / "submissions"
    truth.mkdir(parents=True)
    (truth / "other.csv").write_text("case_id,prediction\n")
    (p["truth_root"] / "tasks").mkdir()
    (p["truth_root"] / "tasks" / "test_labels.csv").write_text("case_id,label\nt1,dir0\n")
    other = p["runs_root"] / "b1" / "other_run" / "a1" / "workspace"
    other.mkdir(parents=True)
    (other / "notes.txt").write_text("x")
    (p["runs_root"] / "b1" / "runs.jsonl").write_text("{}\n")
    p["data_root"].mkdir()
    (p["data_root"] / "raw.csv").write_text("x")
    ws = p["runs_root"] / "b1" / "this_run" / "a1" / "workspace"
    ws.mkdir(parents=True)
    (ws / "TASK.md").write_text("task")

    targets = {
        "read": {"truth_labels": str(p["truth_root"] / "tasks" / "test_labels.csv"), "submission": str(truth / "other.csv"),
                 "other_workspace": str(other / "notes.txt"), "run_records": str(p["runs_root"] / "b1" / "runs.jsonl"),
                 "data_root": str(p["data_root"] / "raw.csv"), "repo": str(REPO_ROOT / "pyproject.toml"),
                 "own_workspace": str(ws / "TASK.md")},
        "list": {"truth_root": str(p["truth_root"]), "runs_root": str(p["runs_root"]), "repo": str(REPO_ROOT)},
        "write": {"truth": str(p["truth_root"] / "planted.csv"), "repo": str(REPO_ROOT / "planted.txt"),
                  "own_workspace": str(ws / "out.txt")},
    }
    wrap = sandbox.wrapper(cfg, ws, network=False)
    res = CLIBackend([sys.executable, "-I", "-c", PROBE, "{prompt}"]).run(json.dumps(targets), ws, timeout_s=30, wrap=wrap)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    allowed = {k for k, v in out.items() if v == "allowed"}
    assert allowed == {"read:own_workspace", "write:own_workspace"}, out
    assert not (p["truth_root"] / "planted.csv").exists() and not (REPO_ROOT / "planted.txt").exists()


def test_truth_root_is_owner_only(dummy):
    from nabench.runner import create_batch
    create_batch(dummy, {"seed": 0, "blocks": [{"tasks": ["dummy_spikes"], "conditions": ["C1"], "models": ["fake-scripted"]}]}, None)
    assert Path(dummy["paths"]["truth_root"]).stat().st_mode & 0o077 == 0
