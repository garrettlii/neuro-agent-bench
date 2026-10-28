"""Run agent CLIs as a separate macOS user (`sandbox.agent_user`) instead of under sandbox-exec.

Agent CLIs sandbox their own commands with sandbox-exec, and sandbox-exec cannot start inside a
profile that denies file access. So a model with `isolation: agent_user` gets no harness sandbox:
file permissions keep the agent user out of this user's home (the repository, data, runs and truth
roots, other projects), and the CLI's own sandbox keeps agent commands offline (checked by the
network probe). Each run's workspace is staged under `sandbox.agent_root`, opened to the agent user
by an ACL for the run, then closed and moved to its place under runs_root.

Nothing the agent writes outside its workspace survives the run: HOME and TMPDIR point to
`sandbox.agent_home`, emptied before and after every run, and afterwards everything the agent user
owns in the shared temp locations, and any file it created in its real home outside Library, is
deleted. Only the sign-in files a model lists in `keep` are carried between runs, through
`sandbox.agent_keep`, which the agent user cannot read; sign-ins a CLI keeps in the keychain persist
in the agent user's own keychain. One-time setup: scripts/setup_agent_user.sh.
"""
from __future__ import annotations

import contextlib
import fcntl
import getpass
import pwd
import shlex
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from .config import REPO_ROOT

FULL = ("read,write,execute,delete,append,readattr,writeattr,readextattr,writeextattr,readsecurity,"
        "list,search,add_file,add_subdirectory,delete_child")
SHARED_TEMP = ("/private/tmp", "/Users/Shared")
KEYCHAIN = "Library/Keychains/nabench.keychain-db"  # in the agent user's real home, empty password (setup_agent_user.sh)


class AgentUserError(RuntimeError):
    pass


def _as(user: str) -> list[str]:
    return [] if user == getpass.getuser() else ["sudo", "-n", "-u", user]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd="/")


def _entry(user: str) -> pwd.struct_passwd:
    try:
        return pwd.getpwnam(user)
    except KeyError:
        raise AgentUserError(f"no user {user!r}; run scripts/setup_agent_user.sh") from None


def prefix(cfg: dict, env: dict[str, str] | None = None) -> list[str]:
    """Runs what follows as the agent user with only these variables; `~/` in env values is agent_home."""
    sb = cfg["sandbox"]
    entry, home = _entry(sb["agent_user"]), sb["agent_home"]
    base = {"HOME": home, "TMPDIR": f"{home}/tmp", "USER": entry.pw_name, "LOGNAME": entry.pw_name,
            "SHELL": entry.pw_shell, "PATH": sb["agent_path"], "LANG": "en_US.UTF-8"}
    extra = {k: home + v[1:] if v.startswith("~/") else v for k, v in (env or {}).items()}
    return [*_as(entry.pw_name), "/usr/bin/env", "-i", *(f"{k}={v}" for k, v in {**base, **extra}.items())]


def kill_command(cfg: dict) -> list[str]:
    """Kills every process of the agent user, including any the agent left running in the background."""
    user = cfg["sandbox"]["agent_user"]
    if user == getpass.getuser():
        raise AgentUserError("the agent user must not be the harness user")
    return [*_as(user), "/usr/bin/pkill", "-KILL", "-U", user]


def sweep(cfg: dict, since: float) -> None:
    """Deletes what the agent user wrote outside agent_home: everything it owns in the shared temp
    locations and its own temp and cache directories, and files created in its real home outside Library."""
    user = cfg["sandbox"]["agent_user"]
    if user == getpass.getuser():
        raise AgentUserError("the agent user must not be the harness user")
    own = [_run([*_as(user), "/usr/bin/getconf", k]).stdout.strip() for k in ("DARWIN_USER_TEMP_DIR", "DARWIN_USER_CACHE_DIR")]
    for root in [*SHARED_TEMP, *filter(None, own)]:
        _run([*_as(user), "/usr/bin/find", root, "-mindepth", "1", "-user", user, "-delete"])
    real = _entry(user).pw_dir
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since))
    # shortcut: Library is left alone (keychain, preferences) and so stays a channel between runs, as do empty directories
    _run([*_as(user), "/usr/bin/find", real, "-mindepth", "1", "-not", "-path", f"{real}/Library/*",
          "(", "-type", "f", "-o", "-type", "l", ")", "-newermt", stamp, "-delete"])


def keychain(cfg: dict) -> None:
    """Makes the agent user's keychain the default for agent_home (macOS looks the default up under $HOME)
    and unlocks it, so CLIs that keep their sign-in in the keychain find it without a dialog."""
    user = cfg["sandbox"]["agent_user"]
    if user == getpass.getuser():
        raise AgentUserError("the agent user must not be the harness user")
    path = f"{_entry(user).pw_dir}/{KEYCHAIN}"
    for args in (["list-keychains", "-d", "user", "-s", path], ["default-keychain", "-d", "user", "-s", path],
                 ["unlock-keychain", "-p", "", path]):
        res = _run([*prefix(cfg), "/usr/bin/security", *args])
        if res.returncode:
            raise AgentUserError(f"keychain {path}: {res.stderr.strip()}; run scripts/setup_agent_user.sh")


def check(cfg: dict) -> None:
    """Refuses unless the agent user runs without a password, cannot enter any protected path, and can reach agent_root."""
    sb, user = cfg["sandbox"], cfg["sandbox"]["agent_user"]
    if _run([*_as(user), "/usr/bin/true"]).returncode:
        raise AgentUserError(f"cannot run commands as {user} without a password; run scripts/setup_agent_user.sh")
    protected = [Path.home(), REPO_ROOT, *(cfg["paths"][k] for k in ("data_root", "runs_root", "truth_root")),
                 sb["agent_keep"], *sb.get("deny_extra", [])]
    protected = [Path(p).expanduser().resolve() for p in protected]
    reachable = [str(p) for p in protected if not _run([*_as(user), "/bin/test", "-e", f"{p}/."]).returncode]
    if reachable:
        raise AgentUserError(f"{user} can enter {reachable}; run scripts/setup_agent_user.sh")
    for key in ("agent_root", "agent_home"):
        path = Path(sb[key]).resolve()
        if any(path.is_relative_to(p) for p in protected) or _run([*_as(user), "/bin/test", "-x", str(path.parent)]).returncode:
            raise AgentUserError(f"{key} {path} must be outside the protected paths and reachable by {user}")


def _wipe(cfg: dict, home: Path) -> None:
    if home.exists():
        _run([*_as(cfg["sandbox"]["agent_user"]), "/usr/bin/find", str(home), "-mindepth", "1", "-delete"])
        home.rmdir()  # works even if the agent closed its home to the harness user


def _make_home(cfg: dict, home: Path, aces: list[str]) -> None:
    """The agent user creates its own home, as macOS will not set a default keychain under a HOME the user does not own.
    It cannot remove or replace the home afterwards: the parent stays writable only by the harness user."""
    user = cfg["sandbox"]["agent_user"]
    grant = f"user:{user} allow add_subdirectory"  # for this mkdir only, so a run cannot leave directories beside its home
    subprocess.run(["/bin/chmod", "+a", grant, str(home.parent)], check=True)
    try:
        for cmd in (["/bin/mkdir", "-m", "700", str(home)], *(["/bin/chmod", "+a", a, str(home)] for a in aces)):
            res = _run([*_as(user), *cmd])
            if res.returncode:
                raise AgentUserError(f"{user} cannot set up {home}: {res.stderr.strip()}; run scripts/setup_agent_user.sh")
    finally:
        subprocess.run(["/bin/chmod", "-a", grant, str(home.parent)], check=True)


def _grant(path: Path, aces: list[str]) -> None:
    for ace in aces:
        subprocess.run(["/bin/chmod", "+a", ace, str(path)], check=True)


@contextmanager
def workspace(cfg: dict, dest: Path, keep: list[str] = ()):
    """Yields a fresh workspace path the agent user may use, with a fresh agent_home holding only the
    `keep` files and a .zprofile that sets PATH. Afterwards kills the agent user's processes, saves the `keep` files, closes the
    workspace to the agent user, moves it to dest and deletes everything else the run wrote. One run at a time."""
    check(cfg)
    sb, me = cfg["sandbox"], getpass.getuser()
    user, root, home, store = sb["agent_user"], Path(sb["agent_root"]), Path(sb["agent_home"]), Path(sb["agent_keep"])
    inherit = [f"user:{user} allow {FULL},file_inherit,directory_inherit", f"user:{me} allow {FULL},file_inherit,directory_inherit"]
    with open(root / ".lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AgentUserError("another agent-user run is in progress") from None
        since, holder = time.time() - 1, None
        try:  # cleans up after a failed setup too, once the home may hold sign-in files
            store.mkdir(mode=0o700, parents=True, exist_ok=True)
            _wipe(cfg, home)
            _make_home(cfg, home, inherit)
            (home / "tmp").mkdir()
            (home / "Library" / "Preferences").mkdir(parents=True)  # security silently skips the keychain settings without it
            # macOS login shells (Codex runs commands in one) put /usr/bin and its Xcode python3 stub ahead of agent_path
            (home / ".zprofile").write_text(f"export PATH={shlex.quote(sb['agent_path'])}\n")
            keychain(cfg)
            for rel in keep:
                (home / rel).parent.mkdir(parents=True, exist_ok=True)  # Codex refuses a CODEX_HOME that does not exist
                if (store / rel).is_file():
                    shutil.copyfile(store / rel, home / rel)
            holder = Path(tempfile.mkdtemp(dir=root))  # mode 0700: only the ACL below lets the agent user in
            # the agent may pass through holder but not rename or delete the workspace inside it
            _grant(holder, [f"user:{user} allow search", *(a + ",only_inherit" for a in inherit)])
            yield holder / "workspace"
        finally:
            subprocess.run(kill_command(cfg), capture_output=True, stdin=subprocess.DEVNULL)
            for rel in keep:  # sign-ins the CLI refreshed during the run
                with contextlib.suppress(OSError):
                    if (home / rel).is_file() and not (home / rel).is_symlink():
                        (store / rel).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        shutil.copyfile(home / rel, store / rel)
            if holder:
                subprocess.run(["/bin/chmod", "-N", str(holder)], check=True)
                if (holder / "workspace").exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(holder / "workspace", dest)
                holder.rmdir()
            _wipe(cfg, home)
            sweep(cfg, since)
