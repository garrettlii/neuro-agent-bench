"""Agent backends. Every backend runs a command template such as
["claude", "-p", "{prompt}", ...] in the run's own fresh workspace, with an
empty stdin, a hard wall-time limit and optional sandbox wrapping.

``cli`` backends shell out to an installed agent CLI. ``scripted`` backends run
a Python file from this package (for example the fake agent) with the same
mechanics, so the harness can be tested end to end without a model call.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import agentuser

INSTRUCTION_FILES = ("CLAUDE.md", "AGENTS.md", "GEMINI.md")


def instruction_files(workdir: str | Path) -> list[Path]:
    """Agent instruction files (read automatically by agent CLIs) inside workdir or in any parent directory."""
    workdir = Path(workdir).resolve()
    found = [p for p in workdir.rglob("*") if p.name in INSTRUCTION_FILES]
    for parent in workdir.parents:
        found += [parent / n for n in INSTRUCTION_FILES if (parent / n).exists()]
    return found


MODEL_USAGE = {"inputTokens": "input_tokens", "cacheReadInputTokens": "cache_read_input_tokens",
               "cacheCreationInputTokens": "cache_creation_input_tokens", "outputTokens": "output_tokens"}


def parse_usage(kind: str | None, stdout: str) -> dict | None:
    """Token use from CLI output: JSON objects (whole output or one per line) with a `usage` dict.
    kind "last" keeps the final one (single result object), its cumulative `modelUsage` summed over models
    when it has one; "sum" adds them up (per-turn events)."""
    # shortcut: assumes the CLI prints usage as {"usage": {...numbers}}; confirm per CLI version before real runs
    if not kind:
        return None
    objs = []
    for chunk in [stdout, *stdout.splitlines()]:
        try:
            objs.append(json.loads(chunk))
        except (json.JSONDecodeError, TypeError):
            pass
    found = [o for o in objs if isinstance(o, dict) and isinstance(o.get("usage"), dict)]
    if not found:
        return None
    models = found[-1].get("modelUsage")
    if kind == "last" and isinstance(models, dict) and models:
        # Claude: `usage` covers only the last turn of a run resumed by a background-task notification;
        # `modelUsage` is the whole run's, per model (D-021)
        return {k: sum(m.get(c, 0) for m in models.values() if isinstance(m, dict)) for c, k in MODEL_USAGE.items()}
    total: dict[str, float] = {}
    for u in [o["usage"] for o in (found[-1:] if kind == "last" else found)]:
        for k, v in u.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                total[k] = total.get(k, 0) + v
    return total


@dataclass
class RunResult:
    returncode: int | None
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool


class CLIBackend:
    name = "cli"

    def __init__(self, command: list[str], env: dict[str, str] | None = None,
                 version_command: list[str] | None = None, usage: str | None = None,
                 kill_command: list[str] | None = None):
        if not any("{prompt}" in part for part in command):
            raise ValueError("CLI command template must contain {prompt}")
        self.command = command
        self.env = {k: os.path.expanduser(v) for k, v in (env or {}).items()}
        self.version_command = version_command
        self.usage = usage
        self.kill_command = kill_command  # for agents running as another user, which killpg cannot reach

    def version(self) -> str | None:
        if not self.version_command:
            return None
        # from /: an agent user cannot enter the harness's working directory, and some CLIs fail without one
        proc = subprocess.run(self.version_command, capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd="/")
        return (proc.stdout or proc.stderr).strip()

    def run(self, prompt: str, workdir: str | Path, timeout_s: float,
            wrap: Callable[[list[str]], list[str]] | None = None) -> RunResult:
        cmd = [part.replace("{prompt}", prompt) for part in self.command]
        if wrap:
            cmd = wrap(cmd)
        start = time.time()
        # empty stdin: some CLIs (codex exec) otherwise wait for extra prompt input on an open stdin;
        # own session so a timeout kills the agent's child processes too
        proc = subprocess.Popen(cmd, cwd=workdir, env={**os.environ, **self.env}, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):  # left only with another user's processes
                os.killpg(proc.pid, signal.SIGKILL)
            if self.kill_command:
                subprocess.run(self.kill_command, capture_output=True, stdin=subprocess.DEVNULL)
            out, err = proc.communicate()
            timed_out = True
        return RunResult(None if timed_out else proc.returncode, out, err, time.time() - start, timed_out)

    def describe(self) -> dict:
        return {"backend": self.name, "command": self.command, "env_vars": sorted(self.env), "usage": self.usage}


class ScriptedBackend(CLIBackend):
    name = "scripted"

    def __init__(self, module: str):
        source = Path(importlib.util.find_spec(module).origin).read_text()
        # source passed inline: the sandbox denies reading the repo, so the file itself is unreachable
        super().__init__([sys.executable, "-I", "-c", source, "{prompt}"], usage="last")
        self.module = module

    def version(self) -> str:
        return f"{self.module} (python {sys.version.split()[0]})"

    def describe(self) -> dict:
        return {"backend": self.name, "module": self.module, "usage": self.usage}


def make_backend(model_cfg: dict, mode: str = "program", cfg: dict | None = None) -> CLIBackend:
    """Native mode needs its own `command_native` template with code execution disabled.
    `isolation: agent_user` runs the CLI, and its version command, as cfg's agent user (nabench.agentuser)."""
    kind = model_cfg["backend"]
    if kind == "cli":
        command = model_cfg["command"] if mode != "native" else model_cfg["command_native"]
        if model_cfg.get("isolation") == "agent_user":
            pre, version = agentuser.prefix(cfg, model_cfg.get("env")), model_cfg.get("version_command")
            return CLIBackend(pre + command, version_command=version and pre + version, usage=model_cfg.get("usage"),
                              kill_command=agentuser.kill_command(cfg))
        return CLIBackend(command, env=model_cfg.get("env"),
                          version_command=model_cfg.get("version_command"), usage=model_cfg.get("usage"))
    if kind == "scripted":
        return ScriptedBackend(model_cfg["module"])
    raise ValueError(f"unknown backend {kind!r}")
