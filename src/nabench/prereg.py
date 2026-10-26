"""Pre-registration guard.

A batch may be generated only if its pre-registration file
1. has no unfilled ``<<FILL`` placeholders,
2. is tracked by git with no uncommitted changes, nor uncommitted or untracked
   files under config/ or tasks/ (task specs are frozen with the pre-registration), and
3. (by default) is contained in a remote-tracking branch, so the push gives an
   external timestamp.
The commit hash, the run plan and the predictions are written into the batch manifest.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

from .config import REPO_ROOT

PLACEHOLDER = "<<FILL"
CLEAN_DIRS = ("config", "tasks")
_YAML_BLOCK = re.compile(r"```yaml\s*\n(.*?)```", re.S)


class PreregError(RuntimeError):
    pass


def _git(args: list[str], repo: Path) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PreregError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def check_prereg(path: str | Path, repo: Path = REPO_ROOT, require_pushed: bool = True,
                 clean_dirs: tuple[str, ...] = CLEAN_DIRS) -> dict:
    repo = Path(repo).resolve()
    path = Path(path)
    full = path if path.is_absolute() else repo / path
    if not full.exists():
        raise PreregError(f"pre-registration file not found: {full}")
    text = full.read_text()
    if PLACEHOLDER in text:
        n = text.count(PLACEHOLDER)
        raise PreregError(f"{path} still has {n} unfilled {PLACEHOLDER} placeholder(s)")
    rel = str(full.resolve().relative_to(repo))
    _git(["ls-files", "--error-unmatch", rel], repo)
    if _git(["status", "--porcelain", "--", rel], repo):
        raise PreregError(f"{rel} has uncommitted changes; commit it first")
    for d in clean_dirs:
        if _git(["status", "--porcelain", "--untracked-files=all", "--", d], repo):
            raise PreregError(f"{d}/ has uncommitted or untracked changes; commit them first")
    commit, when = _git(["log", "-1", "--format=%H|%cI", "--", rel], repo).split("|")
    pushed_to = []
    if require_pushed:
        branches = _git(["branch", "-r", "--contains", commit], repo)
        pushed_to = [b.strip() for b in branches.splitlines() if b.strip()]
        if not pushed_to:
            raise PreregError(f"commit {commit[:10]} with {rel} is not on any remote branch; push it first")
    block = load_block(text)
    return {"path": rel, "commit": commit, "committed_at": when, "pushed_to": pushed_to,
            "plan": block["plan"], "predictions": block["predictions"]}


def load_block(text: str) -> dict:
    """The first fenced ```yaml block with top-level `plan:` and `predictions:` keys."""
    for block in _YAML_BLOCK.findall(text):
        data = yaml.safe_load(block)
        if isinstance(data, dict) and "plan" in data and "predictions" in data:
            return data
    raise PreregError("no ```yaml block with `plan:` and `predictions:` keys found")
