"""D1: the dummy task runs end to end through C1, C2 and C3 with the fake agent, in the sandbox."""
import re
from pathlib import Path

import pytest

from nabench import sandbox
from nabench.batch import batch_dirs, read_jsonl
from nabench.runner import create_batch, run_batch
from nabench.scoring import score_batch

pytestmark = pytest.mark.skipif(not sandbox.available(), reason="no supported sandbox on this platform")
PLAN = {"seed": 0, "blocks": [{"tasks": ["dummy_spikes"], "conditions": ["C1", "C2", "C3"], "models": ["fake-scripted"]}]}


def _normalized(report: str, cond: str) -> tuple[float, float, float]:
    m = re.search(rf"\| agent \| {cond} \| fake-scripted \| 1 \| [\d.]+ \| (-?[\d.]+) \[(-?[\d.]+), (-?[\d.]+)\]", report)
    return tuple(map(float, m.groups()))


def test_dummy_end_to_end(dummy, tmp_path):
    batch = create_batch(dummy, PLAN, None)
    out, final = score_batch(dummy, batch, out_dir=tmp_path / "reports")
    assert not final and "INTERIM" in out.read_text()

    records = run_batch(dummy, batch)
    assert [r["status"] for r in records] == ["ok"] * 3, records
    assert all(r["tokens"] == {"input_tokens": 0, "output_tokens": 0} for r in records)
    assert run_batch(dummy, batch) == []  # nothing left to run

    out, final = score_batch(dummy, batch, out_dir=tmp_path / "reports")
    report = out.read_text()
    assert final and "INTERIM" not in report
    c1, c2, c3 = (_normalized(report, c) for c in ("C1", "C2", "C3"))
    assert c1[0] > 1.0 and 0.5 < c2[0] < c1[0]   # the trained-model tool helps
    assert c3[1] < 0 < c3[2]                     # scrambled signal: at the simple baseline
    assert "C1 - C2 = " in report and "C1 - C3 = " in report

    run_dir, truth_dir = batch_dirs(dummy, batch)
    assert len(list((truth_dir / "scores").glob("*.csv"))) == 6   # per-case scores stay private
    assert "test0000" not in report                               # no case ids in the report
    assert not list(run_dir.rglob("test_labels.csv"))             # truth never staged
    assert all((run_dir / r["run_id"] / "a1" / "logs" / "stdout.txt").exists() for r in records)
    assert len(read_jsonl(truth_dir / "submissions.jsonl")) == 6


def test_error_runs_are_retried_and_keep_report_interim(dummy, tmp_path):
    dummy["models"].append({"name": "broken", "enabled": True, "backend": "cli", "command": ["false", "{prompt}"]})
    batch = create_batch(dummy, {"seed": 0, "blocks": [{"tasks": ["dummy_spikes"], "conditions": ["C1"], "models": ["broken"]}]}, None)
    for _ in range(dummy["run"]["max_attempts"] + 1):
        run_batch(dummy, batch)
    run_dir, _ = batch_dirs(dummy, batch)
    assert [r["attempt"] for r in read_jsonl(run_dir / "runs.jsonl")] == [1, 2, 3]
    out, final = score_batch(dummy, batch, out_dir=tmp_path / "reports")
    assert not final and "1 with errors pending" in out.read_text()
