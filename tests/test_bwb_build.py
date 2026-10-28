"""bwb_neuropixels_wheel data build on a tiny synthetic session."""
import numpy as np
import pytest

from nabench.tasks.bwb_neuropixels_wheel import (N_FRAMES, in_domain, session_samples, window_frames,
                                                 window_spikes)


def test_in_domain_keeps_contained_windows_and_refuses_clipped_ones():
    domain = np.array([[0.0, 10.0], [30.0, 40.0]])
    w = np.array([[0.0, 1.0], [9.0, 10.0], [10.0, 11.0], [20.0, 21.0], [29.0, 30.0], [39.0, 40.0]])
    np.testing.assert_array_equal(in_domain(w, domain), [True, True, False, False, False, True])
    with pytest.raises(ValueError):
        in_domain(np.array([[9.5, 10.5]]), domain)
    with pytest.raises(ValueError):
        in_domain(np.array([[29.5, 30.5]]), domain)


def test_window_spikes_is_half_open_and_relative_to_window_start():
    times = np.array([0.5, 1.0, 1.25, 1.999, 2.0, 3.5])
    units = np.array([0, 1, 2, 3, 4, 5])
    t, u, ptr = window_spikes(times, units, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    np.testing.assert_array_equal(ptr, [0, 3, 4, 4])
    np.testing.assert_array_equal(u, [1, 2, 3, 5])
    np.testing.assert_allclose(t, [0.0, 0.25, 0.999, 0.5], atol=1e-6)
    late, _, _ = window_spikes(np.array([1000.0 + 1 - 1e-9]), np.array([0]), np.array([[1000.0, 1001.0]]))
    assert late.dtype == np.float32 and 0.999 < late[0] < 1.0  # float32 would round it onto the window end


def test_window_frames_snaps_both_edges_up_to_the_grid():
    t0, values = 0.003, np.arange(500.0)
    first = window_frames(values, t0, np.array([[0.5, 1.5], [t0 + 0.2, t0 + 1.2]]))
    assert first.shape == (2, N_FRAMES)
    np.testing.assert_array_equal(first[:, 0], [25, 10])  # ceil((0.5 - t0) * 50) = 25; on-grid start kept
    np.testing.assert_array_equal(np.diff(first, axis=1), 1)
    with pytest.raises(ValueError):
        window_frames(values, t0, np.array([[0.5, 1.0]]))


def test_session_samples_splits_in_time_order_and_drops_gap_trials():
    onsets = np.array([1.0, 3.0, 5.0, 30.0, 32.0, 60.0, 62.0, 64.0])
    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(0, 70, 2000))
    rec = {"times": times, "units": rng.integers(0, 7, times.size), "n_units": 7,
           "windows": np.stack([onsets, onsets + 1.0], 1), "trial_index": np.arange(2, 10),
           "task_domain": np.stack([onsets - 0.5, onsets + 1.5], 1),
           "train_domain": np.array([[0.0, 4.5]]),   # split edges sit on task edges; trial 4 (onset 5.0) is in the gap
           "val_domain": np.array([[29.0, 33.5]]),
           "test_domain": np.array([[59.0, 70.0]]),
           "wheel": np.abs(np.sin(np.arange(3500) / 10)), "wheel_t0": 0.0, "wheel_domain": np.array([[0.0, 70.0]])}
    s = session_samples(rec)
    np.testing.assert_array_equal(s["train"]["trial_index"], [2, 3])
    np.testing.assert_array_equal(s["val"]["trial_index"], [5, 6])
    np.testing.assert_array_equal(s["test"]["trial_index"], [7, 8, 9])
    te = s["test"]
    assert te["wheel_speed"].shape == (3, N_FRAMES)
    np.testing.assert_allclose(te["wheel_speed"][1], rec["wheel"][3100:3150], rtol=1e-6)
    in_trial = (times >= 62.0) & (times < 63.0)
    np.testing.assert_allclose(te["spike_times"][te["spike_ptr"][1]:te["spike_ptr"][2]], times[in_trial] - 62.0, atol=1e-5)
    assert te["spike_ptr"][-1] == len(te["spike_times"]) == len(te["spike_units"])
