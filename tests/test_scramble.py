import numpy as np

from nabench.scramble import permute, phase_randomize


def test_phase_randomize_keeps_amplitude_spectrum():
    rng = np.random.default_rng(0)
    t = np.arange(256)
    x = np.stack([np.sin(t / 5) + rng.normal(0, 0.1, 256), np.cos(t / 9)])
    y = phase_randomize(x, rng)
    assert y.shape == x.shape and np.isrealobj(y)
    assert np.allclose(np.abs(np.fft.rfft(y)), np.abs(np.fft.rfft(x)))
    assert np.allclose(y.mean(-1), x.mean(-1))
    assert not np.allclose(x, y)


def test_permute_keeps_rows():
    a = np.arange(20).reshape(10, 2)
    b = permute(a, np.random.default_rng(0))
    assert sorted(map(tuple, b)) == sorted(map(tuple, a)) and not np.array_equal(a, b)
