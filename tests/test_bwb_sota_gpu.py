"""The GPU SOTA rerun's rules (D-018, D-019): parallel seeds merged into one result, the 4 h job limit, the import check."""
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from nabench.config import REPO_ROOT
from nabench.metrics import write_cases
from nabench.tasks import bwb_neuropixels_wheel as bwb
from nabench.tasks.bwb_sota_gpu import LIMIT_S, Stop, best_epoch, merge_provenance, part_problems, provenance_problems, stop_after

EVAL = (43, 44, 45, 46, 47)
CODE = ("__init__.py", "metrics.py", "tasks/__init__.py", "tasks/bwb_neuropixels_wheel.py", "tasks/bwb_sota_gpu.py")


def _seed(k, start, end, gpu="NVIDIA H100", bf16=True):
    return {"eval_seed": EVAL[k - 1], "device": gpu, "bf16_supported": bf16, "precision": "bf16" if bf16 else "fp32",
            "start_unix": start, "end_unix": end}


def _parts(ends=(3600, 7200)):
    """Two parts run in parallel from t=1000: seeds 1-3 on one GPU, seeds 4-5 on another (fp32)."""
    a = {"export_manifest_sha256": "e", "inputs_sha256": {"x": "1"}, "seeds": {
        "1": _seed(1, 1000, 2000), "2": _seed(2, 2000, 3000), "3": _seed(3, 3000, 1000 + ends[0])}}
    b = {"export_manifest_sha256": "e", "inputs_sha256": {"x": "1"}, "seeds": {
        "4": _seed(4, 1000, 4000, "NVIDIA V100", False), "5": _seed(5, 4000, 1000 + ends[1], "NVIDIA V100", False)}}
    return [a, b]


def test_parallel_parts_merge_with_each_seeds_gpu_and_the_jobs_wall_time():
    parts = _parts()
    assert part_problems(parts, "e") == []
    prov = merge_provenance(parts)
    assert prov["seeds"] == list(EVAL) and prov["per_seed"]["4"]["device"] == "NVIDIA V100"
    assert prov["wall_s"] == 7200 and provenance_problems(prov, EVAL) == []   # first start 1000, last end 8200


def test_parts_that_cannot_merge():
    a, b = _parts()
    assert part_problems([a, b], "other") == ["part 1: made from another export", "part 2: made from another export"]
    assert part_problems([a], "e") == ["seeds run [1, 2, 3], not each of [1, 2, 3, 4, 5] once"]
    assert part_problems([a, b, {**b, "seeds": {"3": a["seeds"]["3"]}}], "e")[0].startswith("seeds run [1, 2, 3, 3, 4, 5]")
    assert part_problems([a, {**b, "inputs_sha256": {"x": "2"}}], "e") == ["parts read different inputs"]


def test_import_refuses_a_job_over_four_hours_and_bad_seed_records():
    prov = merge_provenance(_parts(ends=(3600, LIMIT_S + 1)))
    assert provenance_problems(prov, EVAL)[0].startswith(f"the job took {LIMIT_S + 1} s")
    prov = merge_provenance(_parts())
    assert provenance_problems({**prov, "wall_s": 100}, EVAL) == ["recorded wall time does not match the seeds' times"]
    bad = json.loads(json.dumps(prov))
    bad["per_seed"]["2"].update(precision="fp32")              # bf16 was available
    bad["per_seed"]["5"].update(device="", end_unix=0)
    assert provenance_problems(bad, EVAL) == [
        "seed 2: precision is not bf16 on a GPU that supports it, or fp32 on one that does not",
        "seed 5: no GPU model recorded", "seed 5: no valid start and end time"]
    assert provenance_problems({**prov, "seeds": [43]}, EVAL) == ["not one result per seed for all 5 eval seeds"]
    assert provenance_problems({}, EVAL) == ["not one result per seed for all 5 eval seeds"]


def _job_folders(tmp_path, ends):
    """A minimal export (the job's code and a manifest), two parts as the job writes them, and the task data."""
    exp, data = tmp_path / "export", tmp_path / "data"
    for rel in CODE:
        (exp / "code" / "nabench" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / "src" / "nabench" / rel, exp / "code" / "nabench" / rel)
    bwb.write_manifest(exp, [f"code/nabench/{rel}" for rel in CODE])
    for rel in bwb.SOTA_INPUTS:
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        (data / rel).write_bytes(rel.encode())
    (data / "test" / "case_ids.csv").write_text("case_id\ns01\ns02\n")
    (data / "test" / "trial_index.csv").write_text("case_id,trial_index\ns01,5\ns02,9\ns02,10\n")
    inputs = {rel: bwb._sha256(data / rel) for rel in bwb.SOTA_INPUTS}
    rng, folders = np.random.default_rng(0), []
    for i, part in enumerate(_parts(ends), 1):
        d = tmp_path / f"part{i}"
        (d / "seeds").mkdir(parents=True)
        for k in part["seeds"]:
            write_cases(d / "seeds" / f"seed{k}.csv", {"s01": bwb._fmt(rng.uniform(0, 2, 50)), "s02": bwb._fmt(rng.uniform(0, 2, 100))})
        part.update(export_manifest_sha256=bwb._sha256(exp / "MANIFEST.sha256"), inputs_sha256=inputs)
        (d / "part.json").write_text(json.dumps(part))
        bwb.write_manifest(d, ["part.json", *(f"seeds/seed{k}.csv" for k in part["seeds"])])
        folders.append(str(d))
    return exp, data, folders


def _merge(exp, out, folders):
    return subprocess.run([sys.executable, "-I", str(REPO_ROOT / "scripts" / "sota_gpu_job.py"), str(exp), str(out),
                           "--merge", *folders], capture_output=True, text=True)


def test_merge_writes_the_result_the_batch_expects(tmp_path):
    exp, data, folders = _job_folders(tmp_path, ends=(3600, 7200))
    res = _merge(exp, tmp_path / "out", folders)
    assert res.returncode == 0, res.stderr
    assert bwb.check_sota_result(tmp_path / "out", data) == []   # what sota_baseline checks in the batch
    prov = json.loads((tmp_path / "out" / "provenance.json").read_text())
    assert provenance_problems(prov, EVAL) == [] and prov["wall_s"] == 7200


def test_merge_refuses_a_job_over_four_hours(tmp_path):
    exp, data, folders = _job_folders(tmp_path, ends=(3600, LIMIT_S + 60))
    res = _merge(exp, tmp_path / "out", folders)
    assert res.returncode != 0 and "over the 14400 s limit" in res.stderr and not (tmp_path / "out").exists()


def test_trainval_variant_pieces():
    assert best_epoch([0.1, float("nan"), 0.3, 0.3, 0.2]) == 2   # the first best finite epoch, as poyo_finetune keeps
    r2 = stop_after(3)
    assert [r2("p0", None), r2("p1", None)] == [1.0, 2.0]       # rising: poyo_finetune never starts early stopping
    with pytest.raises(Stop) as done:
        r2("p2", None)
    assert done.value.pred == "p2"                               # the predictions after the 3rd epoch
    a, b = _parts()
    assert part_problems([a, {**b, "variant": "trainval"}], "e") == [
        "part 2: the exploratory train+val variant (D-022), never a SOTA result"]
