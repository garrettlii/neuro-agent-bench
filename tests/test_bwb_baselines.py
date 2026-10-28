"""bwb_neuropixels_wheel baseline helpers on tiny synthetic sessions (torch parts skip without torch)."""
import itertools
import json
import shutil

import numpy as np
import pytest

from nabench.metrics import read_cases, read_predictions, write_cases
from nabench.runner import NotReady
from nabench.tasks import bwb_neuropixels_wheel as bwb
from nabench.tasks.bwb_neuropixels_wheel import (automl_candidates, automl_predict, automl_select, binned_counts,
                                                 load_cases, pooled_r2, session_r2, spike_counts)


def _trials(rng, n, n_units, rate=40):
    out = []
    for _ in range(n):
        k = rng.poisson(rate)
        out.append((np.sort(rng.uniform(0, 1, k)).astype(np.float32), rng.integers(0, n_units, k).astype(np.int32)))
    return out


def test_spike_counts_bin_and_flatten_like_the_benchmark():
    trials = [(np.array([0.0, 0.019, 0.02, 0.5, 0.999], np.float32), np.array([0, 1, 1, 2, 0], np.int32)),
              (np.zeros(0, np.float32), np.zeros(0, np.int32))]
    x = spike_counts(trials, 3, 0.02)
    assert x.shape == (2, 50 * 3) and x[1].sum() == 0
    want = np.zeros((50, 3))
    for t, u in zip(*trials[0]):
        want[int(np.floor(float(t) * (1 / 0.02))), u] += 1   # torch_brain bin_spikes (t * rate, so float32 0.02 is bin 0), then (T, N).flatten()
    np.testing.assert_array_equal(x[0], want.ravel())


def test_session_length_rows_survive_the_csv_readers(tmp_path):
    row = " ".join(["1.2345678"] * 50 * 600)  # 600 test trials, well over csv's default 128 KiB field limit
    write_cases(tmp_path / "p.csv", {"s01": row})
    assert read_cases(tmp_path / "p.csv", "prediction") == {"s01": row}
    assert read_predictions(tmp_path / "p.csv", ["s01"]) == ({"s01": row}, {"file": True, "unknown": 0, "duplicate": 0, "error": None})


def test_pooled_r2_is_the_harness_metric_before_clipping():
    rng = np.random.default_rng(0)
    y, p = rng.uniform(0, 3, (7, 50)), rng.uniform(0, 3, (7, 50))
    good = y + rng.normal(0, 0.3, y.shape)
    fmt = lambda a: " ".join(map(str, a.ravel()))  # noqa: E731
    got = session_r2({"a": fmt(good), "b": fmt(p)}, {"a": fmt(y), "b": fmt(y)})
    np.testing.assert_allclose(got, [pooled_r2(good, y), max(pooled_r2(p, y), 0.0)])


def _workspace(tmp_path, rng, n_units=8, n_dev=110, n_val=30, n_test=20):
    d = tmp_path / "ws" / "data"
    (d / "dev").mkdir(parents=True)
    (d / "test").mkdir()
    w = rng.uniform(0.5, 1.5, n_units)
    dev, test, y = {}, {}, {}
    for cid in ("s01", "s02"):
        for arrs, n in ((dev, n_dev), (test, n_test)):
            tr = _trials(rng, n, n_units)
            arrs |= {f"{cid}_n_units": np.array(n_units), f"{cid}_trial_index": np.arange(n),
                     f"{cid}_spike_times": np.concatenate([t for t, _ in tr]),
                     f"{cid}_spike_units": np.concatenate([u for _, u in tr]),
                     f"{cid}_spike_ptr": np.concatenate([[0], np.cumsum([len(t) for t, _ in tr])])}
            if arrs is dev:
                x = spike_counts(tr, n_units, 0.1).reshape(n, 10, n_units)
                y[cid] = np.repeat(x @ w, 5, axis=1).astype(np.float32)  # speed follows the counts
        dev[f"{cid}_is_val"] = np.arange(n_dev) >= n_dev - n_val
    np.savez(d / "dev" / "sessions.npz", **dev)
    np.savez(d / "dev" / "wheel_speed.npz", **y)
    np.savez(d / "test" / "sessions.npz", **test)
    (d / "test" / "case_ids.csv").write_text("case_id\ns01\ns02\n")
    return d.parent


def test_load_cases_rebuilds_trials(tmp_path):
    cases = load_cases(_workspace(tmp_path, np.random.default_rng(1)))
    assert [c["id"] for c in cases] == ["s01", "s02"]
    c = cases[0]
    assert len(c["dev"]) == 110 and len(c["test"]) == 20 and c["y"].shape == (110, 50) and c["is_val"].sum() == 30
    assert all(len(t) == len(u) and np.all((t >= 0) & (t < 1)) for t, u in c["dev"])


def test_automl_selects_on_val_and_predicts_every_target(tmp_path):
    cases = load_cases(_workspace(tmp_path, np.random.default_rng(2)))
    searched = []
    best = automl_select(cases, wall_time_s=3, searched=searched)   # the search runs until the budget is spent
    assert len(searched) > 20 and all(n == 2 and f == 0 for _, n, f in searched[:20])   # the fixed grid, then random
    # (a few random ones fail here: PCA wider than these 8-unit sessions' features; failures count as not chosen)
    for c in cases:
        name, score = best[c["id"]]
        assert name and score > 0.5   # a linear signal in the counts
        assert automl_predict(c, name, c["test"]).shape == (20, 50)
    assert all(v == ("", -np.inf) for v in automl_select(cases, wall_time_s=-1).values())   # no budget, no fit
    with pytest.raises(ValueError):
        automl_predict(cases[0], "", cases[0]["test"])


def test_automl_random_search_covers_every_family(tmp_path):
    c = load_cases(_workspace(tmp_path, np.random.default_rng(4), n_units=20))[0]   # enough features for PCA 64
    tr = ~c["is_val"]
    seen = {}
    for name, bin_size, make in itertools.islice(automl_candidates(), 20, 400):
        family = name.split(" pca=")[0].split(" ", 1)[1].split("=")[0].rsplit(" ", 1)[0]
        if family not in seen:
            x = binned_counts(c["dev"], c["n_units"], bin_size)
            seen[family] = make().fit(x[tr], c["y"][tr]).predict(x[~tr])
            assert automl_predict(c, name, c["test"]).shape == (20, 50)   # found again by name
    assert set(seen) == {"ridge", "pls", "kernel ridge", "knn", "random forest", "extra trees", "mlp"}
    assert all(p.shape == (30, 50) and np.isfinite(p).all() for p in seen.values())


def _sota_files(tmp_path, n_seeds=5):
    """A task data folder with two cases (2 and 3 test trials) and a job output that passes check_sota_result."""
    data, out = tmp_path / "data", tmp_path / "out"
    for rel in bwb.SOTA_INPUTS:
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        (data / rel).write_bytes(rel.encode())
    (data / "test" / "case_ids.csv").write_text("case_id\ns01\ns02\n")
    (data / "test" / "trial_index.csv").write_text("case_id,trial_index\ns01,5\ns01,6\ns02,9\ns02,10\ns02,11\n")
    rng = np.random.default_rng(0)
    seeds = [{"s01": rng.uniform(0, 2, 100), "s02": rng.uniform(0, 2, 150)} for _ in range(n_seeds)]
    (out / "seeds").mkdir(parents=True)
    for k, p in enumerate(seeds, 1):
        write_cases(out / "seeds" / f"seed{k}.csv", {c: bwb._fmt(v) for c, v in p.items()})
    write_cases(out / "predictions.csv", {c: bwb._fmt(np.mean([p[c] for p in seeds], 0)) for c in ("s01", "s02")})
    (out / "provenance.json").write_text(json.dumps({"inputs_sha256": {r: bwb._sha256(data / r) for r in bwb.SOTA_INPUTS}}))
    bwb.write_manifest(out, bwb.SOTA_RESULT)
    return data, out


def test_sota_result_check_accepts_a_good_job_and_names_each_fault(tmp_path):
    data, out = _sota_files(tmp_path)
    assert bwb.check_sota_result(out, data) == []
    (data / "dev" / "sessions.npz").write_bytes(b"other")   # inputs changed since the export
    assert bwb.check_sota_result(out, data) == ["dev/sessions.npz: the job read a different file"]
    data, out = _sota_files(tmp_path / "b")
    (out / "seeds" / "seed2.csv").write_text("case_id,prediction\ns01,1 2\n")   # changed after the manifest
    assert bwb.check_sota_result(out, data) == ["seeds/seed2.csv: missing or checksum differs"]
    bwb.write_manifest(out, bwb.SOTA_RESULT)
    assert bwb.check_sota_result(out, data) == ["seeds/seed2.csv: its cases are not those in test/case_ids.csv"]
    write_cases(out / "seeds" / "seed2.csv", {"s01": "1 2", "s02": " ".join(["nan"] * 150)})
    bwb.write_manifest(out, bwb.SOTA_RESULT)
    assert bwb.check_sota_result(out, data) == ["seeds/seed2.csv: 2 case(s) without 50 finite values per test trial"]
    data, out = _sota_files(tmp_path / "c")
    write_cases(out / "predictions.csv", {"s01": " ".join(["1"] * 100), "s02": " ".join(["1"] * 150)})
    bwb.write_manifest(out, bwb.SOTA_RESULT)
    assert bwb.check_sota_result(out, data) == ["predictions.csv: not the mean of the seeds for 2 case(s)"]
    assert bwb.check_sota_result(tmp_path / "missing", data)[0].startswith("no readable MANIFEST.sha256")


def test_sota_baseline_submits_the_imported_ensemble(tmp_path, monkeypatch):
    data, out = _sota_files(tmp_path)
    monkeypatch.setattr(bwb, "_task_root", lambda: tmp_path / "task")
    ws = tmp_path / "ws"
    ws.mkdir()
    shutil.copytree(data, ws / "data")
    with pytest.raises(NotReady):   # nothing imported yet: skipped this pass, not an error
        bwb.sota_baseline(ws, {})
    (tmp_path / "task").mkdir()
    out.rename(tmp_path / "task" / "sota_gpu")
    preds = bwb.sota_baseline(ws, {})
    assert preds == read_cases(tmp_path / "task" / "sota_gpu" / "predictions.csv", "prediction")
    assert sorted(p.name for p in (ws / "seeds").iterdir()) == [f"seed{k}.csv" for k in range(1, 6)]
    (ws / "data" / "test" / "sessions.npz").write_bytes(b"other")
    shutil.rmtree(ws / "seeds")
    with pytest.raises(ValueError, match="test/sessions.npz: the job read a different file"):
        bwb.sota_baseline(ws, {})


def test_linear_fit_dense_and_embedding_bag_paths_agree(tmp_path):
    pytest.importorskip("torch")
    from nabench.tasks.bwb_neuropixels_wheel import _pick, linear_fit

    c = load_cases(_workspace(tmp_path, np.random.default_rng(3)))[0]
    tr, va = ~c["is_val"], c["is_val"]
    args = (_pick(c["dev"], tr), c["y"][tr], _pick(c["dev"], va), c["y"][va], c["test"], c["n_units"])
    kw = dict(bin_size=0.1, num_epochs=30, batch_size=16, base_lr=1e-2, seed=5)
    r_dense, p_dense = linear_fit(*args, **kw)
    r_bag, p_bag = linear_fit(*args, **kw, dense_limit=0)
    assert r_dense > 0.5 and p_dense.shape == (20, 50)
    assert abs(r_dense - r_bag) < 1e-3 and np.allclose(p_dense, p_bag, atol=1e-3)
    assert linear_fit(*args, **kw)[0] == r_dense   # deterministic given the seed
    with pytest.raises(ValueError):
        linear_fit(*args, **{**kw, "batch_size": 128})


def test_per_sample_sdpa_is_block_diagonal_attention():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_brain")
    from torch_brain.nn import RotaryTimeEmbedding

    from nabench.tasks.bwb_neuropixels_wheel import per_sample_sdpa

    torch.manual_seed(0)
    H, D = 2, 8
    emb = RotaryTimeEmbedding(head_dim=D, rotate_dim=D // 2, t_min=1e-4, t_max=2.0627)
    for qs, ks in (([4, 4, 4], [6, 6, 6]), ([3, 5], [7, 2])):
        q, k, v = torch.randn(sum(qs), H * D), torch.randn(sum(ks), H * D), torch.randn(sum(ks), H * D)
        got = per_sample_sdpa(query=q, key=k, value=v, q_pos_emb=emb(torch.zeros(sum(qs))), kv_pos_emb=emb(torch.zeros(sum(ks))),
                              q_seqlen=torch.tensor(qs), kv_seqlen=torch.tensor(ks), num_heads=H, dropout_p=0.0,
                              rotate_value=True)   # time 0: the rotations are the identity
        want, a, b = [], 0, 0
        for nq, nk in zip(qs, ks):
            Q, K, V = (x.unflatten(-1, (H, D)) for x in (q[a:a + nq], k[b:b + nk], v[b:b + nk]))
            att = torch.softmax(torch.einsum("qhd,khd->hqk", Q, K) / D ** 0.5, -1)
            want.append(torch.einsum("hqk,khd->qhd", att, V).flatten(-2))
            a, b = a + nq, b + nk
        assert torch.allclose(got, torch.cat(want), atol=1e-5)
