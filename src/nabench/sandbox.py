"""OS sandbox for agent runs (macOS sandbox-exec).

The agent may read and write its own workspace. It may not read or write the
repository (except the virtualenv's files, so the shared Python works), data_root,
runs_root (other workspaces, logs), truth_root, or any extra denied path. Writes
are limited to the workspace, temp directories and configured CLI state
directories. Network is denied unless the model's CLI needs it to reach its own
API, in which case the CLI's own tool sandbox must block network for agent code.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .config import REPO_ROOT

TEMP_WRITE = ("/dev", "/private/tmp", "/private/var/folders")


class SandboxError(RuntimeError):
    pass


def available() -> bool:
    # shortcut: macOS only; on Linux use bubblewrap or a container with the same rules before real runs
    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


def _q(path) -> str:
    p = str(Path(path).expanduser().resolve())
    if '"' in p or "\\" in p:
        raise SandboxError(f"unsupported character in path {p!r}")
    return f'(subpath "{p}")'


def profile(cfg: dict, workspace: Path, network: bool, repo_root: Path = REPO_ROOT) -> str:
    sb = cfg.get("sandbox", {})
    protected = [repo_root, *(cfg["paths"][k] for k in ("data_root", "runs_root", "truth_root")),
                 *sb.get("deny_extra", [])]
    lines = ["(version 1)", "(allow default)",
             '(deny file-write* (subpath "/"))',
             f"(allow file-write* {' '.join(_q(p) for p in [*TEMP_WRITE, *sb.get('extra_write', [])])})",
             f"(deny file-read* file-write* {' '.join(_q(p) for p in protected)})",
             f"(allow file-read* {_q(repo_root / '.venv')})",
             f"(allow file-read* file-write* {_q(workspace)})"]
    # path resolution (realpath, getcwd) needs stat on the ancestors of what is allowed; stat reveals no contents
    ancestors = {a for p in (repo_root / ".venv", workspace) for a in Path(p).resolve().parents}
    lines.append(f"(allow file-read-metadata {' '.join(_q(a).replace('subpath', 'literal') for a in sorted(ancestors))})")
    if not network:
        lines.append("(deny network*)")
    return "\n".join(lines)


def wrapper(cfg: dict, workspace: Path, network: bool):
    if not available():
        raise SandboxError("no supported sandbox on this platform; agent runs are refused")
    prof = profile(cfg, workspace, network)
    return lambda cmd: ["sandbox-exec", "-p", prof, *cmd]
