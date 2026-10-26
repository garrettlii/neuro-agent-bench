"""Configs with data, runs and truth under a temporary directory, and the built dummy task."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from nabench.config import DEFAULT_CONFIG, resolve_paths
from nabench.tasks.dummy_spikes import build


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["paths"] = {k: str(tmp_path / k.replace("_root", "")) for k in ("data_root", "runs_root", "truth_root")}
    return resolve_paths(raw)


@pytest.fixture
def dummy(cfg: dict) -> dict:
    build(Path(cfg["paths"]["data_root"]) / "tasks" / "dummy_spikes", Path(cfg["paths"]["truth_root"]) / "tasks" / "dummy_spikes")
    return cfg


def git(repo, *args):
    env = {**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.com",
           "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.com"}
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)
