"""References: prediction sets made in the baseline step, scored like the baselines, never used in normalization."""
import re
import shutil

import numpy as np
import pytest
import yaml

from nabench import sandbox
from nabench.metrics import write_cases
from nabench.runner import create_batch, run_batch
from nabench.scoring import score_batch
from nabench.spec import TASKS_DIR
from nabench.tasks import bwb_references

REFS = '''from nabench.metrics import read_cases

def signal_free(workspace, budget, baselines):
    return read_cases(baselines["simple"] / "submission" / "predictions.csv", "prediction")

def sota_seed1(workspace, budget, baselines):
    return read_cases(baselines["sota"] / "submission" / "predictions.csv", "prediction")
'''


def test_bwb_profile_and_seed_references(tmp_path):
    data = tmp_path / "data"
    (data / "dev").mkdir(parents=True)
    (data / "test").mkdir()
    y = np.arange(100, dtype=np.float32).reshape(2, 50)
    np.savez(data / "dev" / "wheel_speed.npz", s01=y, s02=y + 1)
    (data / "test" / "case_ids.csv").write_text("case_id\ns01\ns02\n")
    (data / "test" / "trial_index.csv").write_text("case_id,trial_index\ns01,0\ns01,1\ns01,2\ns02,0\n")
    preds = bwb_references.profile(tmp_path, {}, {})
    assert np.allclose(np.array(preds["s01"].split(), float), np.tile(y.mean(0), 3))
    assert len(preds["s02"].split()) == 50
    (tmp_path / "sota" / "seeds").mkdir(parents=True)
    write_cases(tmp_path / "sota" / "seeds" / "seed3.csv", {"s01": "1 2"})
    assert bwb_references.sota_seed3(tmp_path, {}, {"sota": tmp_path / "sota"}) == {"s01": "1 2"}


@pytest.mark.skipif(not sandbox.available(), reason="no supported sandbox on this platform")
def test_references_are_scored_but_not_normalized(dummy, tmp_path, monkeypatch):
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "nab_test_refs.py").write_text(REFS)
    monkeypatch.syspath_prepend(str(tmp_path / "refs"))
    tasks = tmp_path / "tasks"
    shutil.copytree(TASKS_DIR / "dummy_spikes", tasks / "dummy_spikes")
    spec = yaml.safe_load((tasks / "dummy_spikes" / "spec.yaml").read_text())
    spec["references"] = [{"name": "signal_free", "function": "nab_test_refs:signal_free", "leakage_margin": 0.05,
                           "description": "the simple baseline again"},
                          {"name": "sota_seed1", "function": "nab_test_refs:sota_seed1", "description": "SOTA again"}]
    spec["baselines"]["sota"]["published"]["value"] = 0.5
    spec["baselines"]["sota"]["hardware"] = "SOTA elsewhere."
    (tasks / "dummy_spikes" / "spec.yaml").write_text(yaml.safe_dump(spec))
    dummy["models"].append({**next(m for m in dummy["models"] if m["name"] == "fake-scripted"), "name": "fake-b"})
    plan = {"seed": 0, "blocks": [{"tasks": ["dummy_spikes"], "conditions": ["C1", "C2", "C3"], "models": ["fake-scripted", "fake-b"]}]}
    batch = create_batch(dummy, plan, None, tasks_dir=tasks)
    run_batch(dummy, batch)
    out, final = score_batch(dummy, batch, out_dir=tmp_path / "reports")
    report = out.read_text()
    assert final and "11/11 runs and baselines scored" in report  # 6 runs, 3 baselines, 2 references
    assert "| normalized vs SOTA | normalized vs published SOTA |" in report
    assert "). Hardware: SOTA elsewhere.\n" in report
    assert re.search(r"\| reference \| - \| signal_free \| 1 \| [\d.]+ \| 0.000 \[0.000, 0.000\] \| 0.000 \[0.000, 0.000\] \|", report)
    pub_col = re.search(r"\| baseline \| - \| sota \| 1 \| [\d.]+ \| 1.000 \[1.000, 1.000\] \| (-?[\d.]+) \[", report)
    assert pub_col and pub_col.group(1) != "1.000"  # the published 0.5 differs from the rerun
    sota = re.search(r"SOTA rerun \(.*?\): ([\d.]+)", report).group(1)
    assert f"| reference | - | sota_seed1 | 1 | {sota} | 1.000 [1.000, 1.000] |" in report
    assert f"Mean of the 1 per-seed SOTA scores (raw, over cases and seeds): {sota} [" in report
    assert "- fake-scripted: C2 - automl = " in report and "- C1: fake-scripted - fake-b = " in report
    assert "- C2: fake-scripted - fake-b = " in report
    assert re.search(r"- fake-b: C3 lower CI bound -?[\d.]+, (not )?above 0; C3 - signal_free = -?[\d.]+, "
                     r"(not )?more than 0.05: (TRIGGERED|not triggered)", report)


LATER = '''from pathlib import Path

from nabench.metrics import read_cases
from nabench.runner import NotReady


def later(workspace, budget, baselines):
    if not Path(__file__).with_name("ready").exists():
        raise NotReady("not yet")
    return read_cases(baselines["simple"] / "submission" / "predictions.csv", "prediction")
'''


@pytest.mark.skipif(not sandbox.available(), reason="no supported sandbox on this platform")
def test_a_reference_not_ready_is_skipped_and_filled_on_a_later_pass(dummy, tmp_path, monkeypatch):
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "nab_test_later.py").write_text(LATER)
    monkeypatch.syspath_prepend(str(tmp_path / "refs"))
    tasks = tmp_path / "tasks"
    shutil.copytree(TASKS_DIR / "dummy_spikes", tasks / "dummy_spikes")
    spec = yaml.safe_load((tasks / "dummy_spikes" / "spec.yaml").read_text())
    spec["references"] = [{"name": "later", "function": "nab_test_later:later", "description": "ready later"}]
    (tasks / "dummy_spikes" / "spec.yaml").write_text(yaml.safe_dump(spec))
    plan = {"seed": 0, "blocks": [{"tasks": ["dummy_spikes"], "conditions": ["C2"], "models": ["fake-scripted"]}]}
    batch = create_batch(dummy, plan, None, tasks_dir=tasks)
    assert [r["status"] for r in run_batch(dummy, batch)] == ["ok"]   # the agent run goes ahead
    out, final = score_batch(dummy, batch, out_dir=tmp_path / "reports")
    assert not final and "1 of 5 planned runs and baselines have no scored submission" in out.read_text()
    (tmp_path / "refs" / "ready").touch()
    assert run_batch(dummy, batch) == []
    out, final = score_batch(dummy, batch, out_dir=tmp_path / "reports")
    assert final and "| reference | - | later | 1 |" in out.read_text()
