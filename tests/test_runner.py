from pathlib import Path

import pytest

from nabench.batch import LedgerError, batch_dirs, read_json, read_jsonl
from nabench.runner import RunError, create_batch, expand_plan, stage, submit
from nabench.spec import TASKS_DIR, SpecError, load_spec, resolve

PLAN = {"seed": 0, "blocks": [{"tasks": ["dummy_spikes"], "conditions": ["C1", "C2"], "models": ["fake-scripted"]}]}
PREREG = {"path": "prereg/x.md", "commit": "abc123", "committed_at": "2026-10-26", "pushed_to": ["origin/main"],
          "plan": PLAN, "predictions": {}}


def test_one_batch_per_preregistration(dummy):
    create_batch(dummy, PLAN, PREREG)
    with pytest.raises(LedgerError, match="already generated"):
        create_batch(dummy, PLAN, PREREG)
    create_batch(dummy, PLAN, PREREG, reason="hardware failure")
    ledger = read_jsonl(Path(dummy["paths"]["truth_root"]) / "ledger.jsonl")
    assert [e["additional_batch_reason"] for e in ledger] == [None, "hardware failure"]


def test_manifest_records_checksums_and_models(dummy):
    batch = create_batch(dummy, PLAN, PREREG)
    run_dir, truth_dir = batch_dirs(dummy, batch)
    m = read_json(run_dir / "manifest.json")
    assert m["prereg"]["commit"] == "abc123" and len(m["runs"]) == 2
    assert all(len(v) == 64 for v in m["inputs_sha256"]["dummy_spikes"].values())
    assert m["models"]["fake-scripted"]["version"].startswith("nabench.fake_agent")
    assert "truth_sha256" not in m and "truth_sha256" in read_json(truth_dir / "manifest.json")


def test_batch_refusals(dummy, tmp_path):
    with pytest.raises(SpecError, match="unfilled"):  # pilot specs still have FILL fields
        create_batch(dummy, {"blocks": [{"tasks": ["nsd_cross_subject_id"], "conditions": ["C1"], "models": ["fake-scripted"]}]}, PREREG)
    real = tmp_path / "tasks" / "dummy_spikes"
    real.mkdir(parents=True)
    (real / "spec.yaml").write_text((TASKS_DIR / "dummy_spikes" / "spec.yaml").read_text().replace("synthetic: true", "synthetic: false"))
    with pytest.raises(RunError, match="not synthetic"):
        create_batch(dummy, PLAN, None, tasks_dir=tmp_path / "tasks")
    with pytest.raises(RunError, match="not configured or not enabled"):
        create_batch(dummy, {"blocks": [{"tasks": ["dummy_spikes"], "conditions": ["C1"], "models": ["opus-5.5-cli"]}]}, None)
    with pytest.raises(RunError, match="submission limit"):
        create_batch(dummy, {"blocks": [{"tasks": ["dummy_spikes"], "conditions": ["C1"], "models": ["fake-scripted"], "reps": 2}]}, None)


def test_workspace_contents_by_condition(dummy, tmp_path):
    spec = resolve(load_spec("dummy_spikes"), dummy)
    conds = dummy["conditions"]
    stage(spec, dummy, conds["C1"], tmp_path / "c1", seed=1)
    stage(spec, dummy, conds["C2"], tmp_path / "c2", seed=1)
    stage(spec, dummy, conds["C3"], tmp_path / "c3", seed=1)
    files = lambda d: sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())
    assert files(tmp_path / "c2") == ["TASK.md", "data/dev/labels.csv", "data/dev/spikes.npy",
                                      "data/test/case_ids.csv", "data/test/spikes.npy"]
    assert "tools/pretrained_decoder/weights.npz" in files(tmp_path / "c1")
    assert "pretrained_decoder" in (tmp_path / "c1" / "TASK.md").read_text()
    assert "pretrained_decoder" not in (tmp_path / "c2" / "TASK.md").read_text()
    assert (tmp_path / "c1" / "TASK.md").read_text() == (tmp_path / "c3" / "TASK.md").read_text()
    assert (tmp_path / "c3" / "data/dev/labels.csv").read_text() != (tmp_path / "c1" / "data/dev/labels.csv").read_text()
    for d in ("c1", "c2", "c3"):
        assert not any("label" in p.name for p in (tmp_path / d / "data" / "test").iterdir())
    with pytest.raises(RunError, match="fresh workspace"):
        stage(spec, dummy, conds["C1"], tmp_path / "c1", seed=1)


def test_submit_limit_and_symlink_refusal(dummy, tmp_path):
    batch = create_batch(dummy, PLAN, None)
    _, truth_dir = batch_dirs(dummy, batch)
    run = expand_plan(PLAN)[0]
    ws = tmp_path / "ws"
    (ws / "submission").mkdir(parents=True)
    secret = Path(dummy["paths"]["truth_root"]) / "tasks" / "dummy_spikes" / "test_labels.csv"
    (ws / "submission" / "predictions.csv").symlink_to(secret)
    assert submit(truth_dir, run, ws, limit=2)["file"] is False
    (ws / "submission" / "predictions.csv").unlink()
    (ws / "submission").rmdir()
    (ws / "submission").symlink_to(secret.parent)  # directory link to the truth folder
    assert submit(truth_dir, run, ws, limit=2)["file"] is False
    with pytest.raises(RunError, match="submission limit"):
        submit(truth_dir, run, ws, limit=2)
