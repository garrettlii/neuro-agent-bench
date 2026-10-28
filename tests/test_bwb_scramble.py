"""bwb_neuropixels_wheel C3 scramble on tiny synthetic sessions."""
import shutil

import numpy as np

from nabench.tasks.bwb_neuropixels_wheel import circular_shift, scramble, window_spikes


def test_circular_shift_equals_shift_modulo_duration_then_sort():
    rng = np.random.default_rng(1)
    times, units = np.sort(rng.uniform(10, 110, 500)), rng.integers(0, 9, 500)
    t, u = circular_shift(times, units, np.array([10.0, 110.0]), 37.3)
    want = 10 + (times - 10 + 37.3) % 100
    order = np.argsort(want, kind="stable")
    np.testing.assert_allclose(t, want[order], atol=1e-9)
    np.testing.assert_array_equal(u, units[order])
    assert np.all(np.diff(t) >= 0)


def _session(rng, start, n_dev, n_test):
    times = np.sort(rng.uniform(start, start + 200, 4000))
    units = rng.integers(0, 6, times.size).astype(np.int32)
    onsets = np.sort(rng.choice(np.arange(start + 5, start + 190, 2.0), n_dev + n_test, replace=False))
    w = np.stack([onsets, onsets + 1.0], 1)
    return {"times": times, "units": units, "span": np.array([start, start + 200.0]),
            "dev_windows": w[:n_dev], "test_windows": w[n_dev:]}


def _stage(tmp_path, rng):
    root, data = tmp_path / "data_root" / "tasks" / "bwb", tmp_path / "ws" / "data"
    for d in ("dev", "test", "scramble"):
        (root / d).mkdir(parents=True)
    dev, test, y = {}, {}, {}
    for cid, start in (("s01", 100.0), ("s02", 5000.0)):
        s = _session(rng, start, 6, 4)
        np.savez(root / "scramble" / f"{cid}.npz", **s)
        for arrs, w, first in ((dev, s["dev_windows"], 0), (test, s["test_windows"], 6)):
            t, u, ptr = window_spikes(s["times"], s["units"], w)
            arrs |= {f"{cid}_n_units": np.array(6), f"{cid}_trial_index": np.arange(first, first + len(w)),
                     f"{cid}_spike_times": t, f"{cid}_spike_units": u, f"{cid}_spike_ptr": ptr}
        dev[f"{cid}_is_val"] = np.arange(6) >= 4
        y[cid] = rng.uniform(0, 5, (6, 50)).astype(np.float32)
    np.savez(root / "dev" / "sessions.npz", **dev)
    np.savez(root / "dev" / "wheel_speed.npz", **y)
    np.savez(root / "test" / "sessions.npz", **test)
    (root / "test" / "case_ids.csv").write_text("case_id\ns01\ns02\n")
    shutil.copytree(root / "dev", data / "dev")
    shutil.copytree(root / "test", data / "test")
    return root, data, {"data": {"dev_inputs": [str(root / "dev" / "sessions.npz")]}}


def test_scramble_recuts_every_window_from_one_shifted_train_and_keeps_labels(tmp_path):
    root, data, spec = _stage(tmp_path, np.random.default_rng(0))
    labels = (data / "dev" / "wheel_speed.npz").read_bytes()
    scramble(data, np.random.default_rng(7), spec)
    assert (data / "dev" / "wheel_speed.npz").read_bytes() == labels
    draws = np.random.default_rng(7)
    for cid in ("s01", "s02"):
        src = np.load(root / "scramble" / f"{cid}.npz")
        start, end = src["span"]
        off = draws.uniform(0.1, 0.9) * (end - start)
        shifted = start + (src["times"] - start + off) % (end - start)
        order = np.argsort(shifted, kind="stable")
        for d in ("dev", "test"):
            old, new = np.load(root / d / "sessions.npz"), np.load(data / d / "sessions.npz")
            for k in ("n_units", "trial_index") + (("is_val",) if d == "dev" else ()):
                np.testing.assert_array_equal(new[f"{cid}_{k}"], old[f"{cid}_{k}"])
            t, u, ptr = window_spikes(shifted[order], src["units"][order], src[f"{d}_windows"])
            np.testing.assert_array_equal(new[f"{cid}_spike_ptr"], ptr)
            np.testing.assert_array_equal(new[f"{cid}_spike_units"], u)
            np.testing.assert_allclose(new[f"{cid}_spike_times"], t, atol=1e-5)
            assert not np.array_equal(new[f"{cid}_spike_units"], old[f"{cid}_spike_units"])


def test_scramble_offset_range_and_determinism(tmp_path):
    root, data, spec = _stage(tmp_path, np.random.default_rng(0))
    pristine = tmp_path / "pristine"
    shutil.copytree(data, pristine)

    class Edge:  # the lowest and highest draws the rule allows
        def __init__(self, x):
            self.x, self.calls = x, []

        def uniform(self, lo, hi):
            self.calls.append((lo, hi))
            return self.x

    for x in (0.1, 0.9):
        shutil.rmtree(data)
        shutil.copytree(pristine, data)
        rng = Edge(x)
        scramble(data, rng, spec)
        assert rng.calls == [(0.1, 0.9)] * 2
        src = np.load(root / "scramble" / "s01.npz")
        t, u = circular_shift(src["times"], src["units"], src["span"], x * 200.0)
        np.testing.assert_array_equal(np.load(data / "test" / "sessions.npz")["s01_spike_units"],
                                      window_spikes(t, u, src["test_windows"])[1])
    runs = []
    for seed in (3, 3, 4):
        shutil.rmtree(data)
        shutil.copytree(pristine, data)
        scramble(data, np.random.default_rng(seed), spec)
        runs.append(np.load(data / "dev" / "sessions.npz")["s02_spike_times"])
    assert np.array_equal(runs[0], runs[1]) and not np.array_equal(runs[0], runs[2])
