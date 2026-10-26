"""Batch bookkeeping. Workspaces, logs and run records go under runs_root/<batch>;
submissions, per-case scores and the ledger go under truth_root. Existing
batches are never overwritten, and one batch is allowed per pre-registration
commit unless a reason is recorded in the ledger."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import secrets
from pathlib import Path


class LedgerError(RuntimeError):
    pass


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def new_batch_id(prefix: str) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{secrets.token_hex(2)}"


def batch_dirs(cfg: dict, batch_id: str, create: bool = False) -> tuple[Path, Path]:
    run_dir = Path(cfg["paths"]["runs_root"]) / batch_id
    truth_dir = Path(cfg["paths"]["truth_root"]) / batch_id
    if create:
        for d in (run_dir, truth_dir):
            if d.exists():
                raise FileExistsError(f"{d} already exists; batches are never regenerated in place")
            d.mkdir(parents=True)
        # scorer-only: owner access, so a run under another user cannot read it either
        Path(cfg["paths"]["truth_root"]).chmod(0o700)
        truth_dir.chmod(0o700)
    return run_dir, truth_dir


def ledger_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["truth_root"]) / "ledger.jsonl"


def check_ledger(cfg: dict, prereg: dict | None, reason: str | None) -> None:
    """One batch per pre-registration commit unless a reason is given."""
    if prereg is None:
        return
    earlier = [e for e in read_jsonl(ledger_path(cfg))
               if e["prereg_path"] == prereg["path"] and e["prereg_commit"] == prereg["commit"]]
    if earlier and not reason:
        raise LedgerError(f"{len(earlier)} batch(es) already generated under this pre-registration "
                          f"({', '.join(e['batch_id'] for e in earlier)}). Pass --additional-batch-reason "
                          "to record why another is needed, or write a new pre-registration.")


def record_batch(cfg: dict, batch_id: str, created_at: str, prereg: dict | None, reason: str | None) -> None:
    append_jsonl(ledger_path(cfg), {
        "batch_id": batch_id, "created_at": created_at, "smoke": prereg is None,
        "prereg_path": prereg and prereg["path"], "prereg_commit": prereg and prereg["commit"],
        "additional_batch_reason": reason})


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def checksums(paths) -> dict[str, str]:
    """SHA-256 of every file, expanding directories."""
    out = {}
    for p in map(Path, paths):
        for f in sorted(p.rglob("*")) if p.is_dir() else [p]:
            if f.is_file():
                out[str(f)] = sha256_file(f)
    return out


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def append_jsonl(path: Path, obj) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]
