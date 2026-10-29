import json
import os
import sys
import time

import pytest

from nabench.backends import CLIBackend, ScriptedBackend, instruction_files, make_backend, parse_usage


def test_empty_stdin_and_prompt_substitution(tmp_path):
    cmd = [sys.executable, "-c", "import sys; print(repr(sys.stdin.read()), sys.argv[1])", "{prompt}"]
    res = CLIBackend(cmd).run("hello", tmp_path, timeout_s=10)
    assert res.returncode == 0 and res.stdout.strip() == "'' hello"


def test_command_needs_prompt_placeholder():
    with pytest.raises(ValueError):
        CLIBackend(["echo", "x"])


def test_hard_timeout_kills_child_processes(tmp_path):
    pidfile = tmp_path / "child.pid"
    code = ("import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            f"open({str(pidfile)!r}, 'w').write(str(p.pid)); time.sleep(60)")
    start = time.time()
    res = CLIBackend([sys.executable, "-c", code, "{prompt}"]).run("x", tmp_path, timeout_s=2)
    assert res.timed_out and res.returncode is None and time.time() - start < 10
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_timeout_runs_kill_command(tmp_path):
    marker = tmp_path / "killed"
    b = CLIBackend([sys.executable, "-c", "import time; time.sleep(60)", "{prompt}"], kill_command=["touch", str(marker)])
    assert b.run("x", tmp_path, timeout_s=1).timed_out and marker.exists()


def test_version_and_command_recorded():
    m = {"backend": "cli", "command": ["agent", "{prompt}"], "version_command": [sys.executable, "--version"],
         "env": {"X_HOME": "~/x"}, "usage": "last"}
    b = make_backend(m)
    assert b.version().startswith("Python")
    assert b.describe() == {"backend": "cli", "command": ["agent", "{prompt}"], "env_vars": ["X_HOME"], "usage": "last"}
    with pytest.raises(KeyError):
        make_backend(m, mode="native")  # native mode needs its own no-code command
    assert ScriptedBackend("nabench.fake_agent").describe()["module"] == "nabench.fake_agent"


def test_parse_usage():
    claude = '{"result": "ok", "usage": {"input_tokens": 10, "output_tokens": 3, "server_tool_use": {"x": 1}}}'
    assert parse_usage("last", claude) == {"input_tokens": 10, "output_tokens": 3}
    codex = "\n".join(['{"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 1}}'] * 3)
    assert parse_usage("sum", codex) == {"input_tokens": 15, "output_tokens": 3}
    assert parse_usage("sum", "no json") is None and parse_usage(None, claude) is None


def test_parse_usage_of_a_resumed_claude_run():
    # resumed by a background-task notification: `usage` is the last turn only, `modelUsage` the whole run
    out = json.dumps({"num_turns": 1, "usage": {"input_tokens": 4, "cache_read_input_tokens": 56608,
                                                "cache_creation_input_tokens": 1204, "output_tokens": 68},
                      "modelUsage": {"claude-opus-5-5": {"inputTokens": 92, "outputTokens": 15522, "cacheReadInputTokens": 1743462,
                                                         "cacheCreationInputTokens": 57812, "costUSD": 1.12, "contextWindow": 1000000},
                                     "claude-haiku-5-5": {"inputTokens": 8, "outputTokens": 2, "cacheReadInputTokens": 0,
                                                          "cacheCreationInputTokens": 0, "costUSD": 0.01}}})
    assert parse_usage("last", out) == {"input_tokens": 100, "cache_read_input_tokens": 1743462,
                                        "cache_creation_input_tokens": 57812, "output_tokens": 15524}


def test_instruction_files_found_in_workdir_and_parents(tmp_path):
    ws = tmp_path / "a" / "ws"
    ws.mkdir(parents=True)
    assert instruction_files(ws) == []
    (tmp_path / "a" / "AGENTS.md").write_text("x")
    (ws / "CLAUDE.md").write_text("x")
    assert {p.name for p in instruction_files(ws)} == {"AGENTS.md", "CLAUDE.md"}
