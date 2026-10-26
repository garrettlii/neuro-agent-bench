import pytest
import yaml

from nabench.config import DEFAULT_CONFIG, REPO_ROOT, load_config, resolve_paths


def _raw(**paths):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["paths"].update(paths)
    return raw


def test_default_config_loads_and_is_outside_repo():
    cfg = load_config()
    for key in ("data_root", "runs_root", "truth_root"):
        assert not cfg["paths"][key].startswith(str(REPO_ROOT))


def test_paths_inside_repo_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="inside the repository"):
        resolve_paths(_raw(runs_root=str(REPO_ROOT / "runs")))


def test_roots_must_not_nest(tmp_path):
    with pytest.raises(ValueError, match="truth_root must not be inside"):
        resolve_paths(_raw(data_root=str(tmp_path / "d"), runs_root=str(tmp_path / "r"), truth_root=str(tmp_path / "r" / "t")))


def test_tokens_are_substituted(tmp_path):
    cfg = resolve_paths(_raw(data_root=str(tmp_path / "d"), runs_root=str(tmp_path / "r"), truth_root=str(tmp_path / "t")))
    assert cfg["paths"]["data_root"] == str((tmp_path / "d").resolve())
