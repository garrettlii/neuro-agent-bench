import getpass
import os
import subprocess
from pathlib import Path

import pytest

from nabench import agentuser
from nabench.batch import batch_dirs, read_jsonl
from nabench.runner import create_batch, run_batch


@pytest.fixture
def as_self(dummy, tmp_path, monkeypatch):
    """The current user stands in for the agent user; the permission check and the user-wide kill and sweep are stubbed."""
    (tmp_path / "agent").mkdir()
    dummy["sandbox"].update(agent_user=getpass.getuser(), agent_root=str(tmp_path / "agent"), agent_path=os.environ["PATH"],
                            agent_home=str(tmp_path / "home"), agent_keep=str(tmp_path / "keep"))
    monkeypatch.setattr(agentuser, "check", lambda cfg: None)
    monkeypatch.setattr(agentuser, "kill_command", lambda cfg: ["/usr/bin/true"])
    monkeypatch.setattr(agentuser, "sweep", lambda cfg, since: None)
    monkeypatch.setattr(agentuser, "keychain", lambda cfg: None)
    return dummy


def test_prefix_clears_the_environment(cfg):
    cfg["sandbox"].update(agent_user=getpass.getuser(), agent_path="/usr/bin:/bin", agent_home="/run/home")
    pre = agentuser.prefix(cfg, {"CODEX_HOME": "~/.codex-bench"})
    assert pre[:2] == ["/usr/bin/env", "-i"] and "PATH=/usr/bin:/bin" in pre
    assert {"HOME=/run/home", "TMPDIR=/run/home/tmp", "CODEX_HOME=/run/home/.codex-bench"} <= set(pre)
    cfg["sandbox"]["agent_user"] = "no-such-user-nabench"
    with pytest.raises(agentuser.AgentUserError, match="setup_agent_user"):
        agentuser.prefix(cfg)


def test_check_refuses_a_user_that_can_enter_protected_paths(cfg, tmp_path):
    cfg["sandbox"].update(agent_user=getpass.getuser(), agent_root=str(tmp_path))
    with pytest.raises(agentuser.AgentUserError, match="can enter"):
        agentuser.check(cfg)
    for user_wide in (agentuser.kill_command, lambda c: agentuser.sweep(c, 0), agentuser.keychain):
        with pytest.raises(agentuser.AgentUserError, match="must not be the harness user"):
            user_wide(cfg)


def test_workspace_is_closed_and_moved(as_self, tmp_path):
    dest = tmp_path / "runs" / "r" / "a1" / "workspace"
    home = tmp_path / "home"
    (tmp_path / "keep" / ".cli").mkdir(parents=True)
    (tmp_path / "keep" / ".cli" / "auth.json").write_text("old")
    with agentuser.workspace(as_self, dest, keep=[".cli/auth.json", ".new/absent.json"]) as ws:
        assert (home / ".cli" / "auth.json").read_text() == "old" and (home / "tmp").is_dir() and (home / ".new").is_dir()
        assert (home / ".zprofile").read_text() == f"export PATH={as_self['sandbox']['agent_path']}\n"
        (home / ".cli" / "auth.json").write_text("refreshed")
        (home / ".cli" / "memory.md").write_text("notes for the next run")
        (ws / "submission").mkdir(parents=True)
        (ws / "submission" / "x").write_text("1")
        assert "allow" in subprocess.run(["ls", "-led", str(ws)], capture_output=True, text=True).stdout
        with pytest.raises(agentuser.AgentUserError, match="in progress"):
            with agentuser.workspace(as_self, tmp_path / "other"):
                pass
    assert (dest / "submission" / "x").read_text() == "1"
    assert [p.name for p in (tmp_path / "agent").iterdir()] == [".lock"]  # holder removed
    assert not home.exists() and sorted(p.name for p in (tmp_path / "keep" / ".cli").iterdir()) == ["auth.json"]
    assert (tmp_path / "keep" / ".cli" / "auth.json").read_text() == "refreshed"
    with pytest.raises(RuntimeError):
        with agentuser.workspace(as_self, tmp_path / "failed") as ws:
            ws.mkdir()
            raise RuntimeError("agent run failed")
    assert (tmp_path / "failed").is_dir()  # moved even when the run raised


def test_failed_setup_still_removes_the_home(as_self, tmp_path, monkeypatch):
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "auth.json").write_text("secret")

    def broken(path, aces):
        raise OSError("no ACL")
    monkeypatch.setattr(agentuser, "_grant", broken)  # fails after the sign-in file is seeded
    with pytest.raises(OSError, match="no ACL"):
        with agentuser.workspace(as_self, tmp_path / "ws", keep=["auth.json"]):
            pass
    assert not (tmp_path / "home").exists() and (tmp_path / "keep" / "auth.json").read_text() == "secret"
    assert "add_subdirectory" not in subprocess.run(["ls", "-led", str(tmp_path)], capture_output=True, text=True).stdout


def test_agent_user_run_end_to_end(as_self, monkeypatch):
    monkeypatch.setenv("NAB_HARNESS_ONLY", "1")
    script = ("env > submission/env.txt; { echo case_id,prediction; tail -n +2 data/test/case_ids.csv | sed 's/$/,a/'; }"
              " > submission/predictions.csv")
    as_self["models"].append({"name": "isolated", "enabled": True, "backend": "cli", "isolation": "agent_user",
                              "command": ["/bin/sh", "-c", script, "{prompt}"], "env": {"X_HOME": "~/x"}})
    batch = create_batch(as_self, {"seed": 0, "blocks": [{"tasks": ["dummy_spikes"], "conditions": ["C2"], "models": ["isolated"]}]}, None)
    [rec] = run_batch(as_self, batch)
    run_dir, truth_dir = batch_dirs(as_self, batch)
    assert rec["status"] == "ok" and rec["submitted"] is True
    env = (run_dir / rec["run_id"] / "a1" / "workspace" / "submission" / "env.txt").read_text()
    assert f"X_HOME={as_self['sandbox']['agent_home']}/x" in env and "NAB_HARNESS_ONLY" not in env
    assert read_jsonl(truth_dir / "submissions.jsonl")[-1]["file"] is True
