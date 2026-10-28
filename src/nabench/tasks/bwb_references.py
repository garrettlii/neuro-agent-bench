"""Reference prediction sets for bwb_neuropixels_wheel: scored next to the baselines, never used in normalization.

Each is called as fn(workspace, budget, baseline_workspaces) after the baselines have run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..metrics import read_cases
from ..runner import NotReady


def profile(workspace: Path, budget: dict, baselines: dict[str, Path]) -> dict[str, str]:
    """Signal-free: each session's mean dev wheel-speed time course after movement onset, repeated for every test
    trial. Uses no neural data."""
    data = workspace / "data"
    dev = np.load(data / "dev" / "wheel_speed.npz")
    trials = np.loadtxt(data / "test" / "trial_index.csv", delimiter=",", skiprows=1, dtype=str, ndmin=2)[:, 0]
    return {c: " ".join(f"{v:.9g}" for v in np.tile(dev[c].mean(0), int((trials == c).sum())))
            for c in read_cases(data / "test" / "case_ids.csv", "case_id")}


def _seed(k: int):
    def sota_seed(workspace: Path, budget: dict, baselines: dict[str, Path]) -> dict[str, str]:
        path = baselines["sota"] / "seeds" / f"seed{k}.csv"
        if not path.exists():
            raise NotReady("the SOTA baseline has not run yet")
        return read_cases(path, "prediction")
    sota_seed.__doc__ = f"POYO+ finetuning seed {k} alone, as imported by sota_baseline (which submits the 5-seed ensemble)."
    return sota_seed


sota_seed1, sota_seed2, sota_seed3, sota_seed4, sota_seed5 = (_seed(k) for k in range(1, 6))
