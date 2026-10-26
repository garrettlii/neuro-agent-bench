"""Scrambled-signal controls (condition C3). Any skill above the simple baseline
on scrambled data means the agent got the answer from somewhere other than the
signal: leakage, memorized labels or a scoring bug."""
from __future__ import annotations

import numpy as np


def permute(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rows in random order (breaks the link between cases and their inputs or labels)."""
    return a[rng.permutation(len(a))]


def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same amplitude spectrum along the last axis, random phases shared across
    leading axes (channels), so per-channel spectra and cross-spectra magnitudes are kept."""
    n = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    phases = rng.uniform(0, 2 * np.pi, spec.shape[-1])
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0  # Nyquist bin must stay real
    return np.fft.irfft(spec * np.exp(1j * phases), n=n, axis=-1)
