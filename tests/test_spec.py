import copy

import pytest

from nabench.spec import SpecError, fills, frozen, load_spec, task_ids, validate

PILOTS = {"niclip_map_to_paper", "nsd_cross_subject_id", "stop_signal_behaviour", "bwb_neuropixels_wheel", "t15_speech"}


def test_every_spec_validates():
    ids = task_ids()
    assert PILOTS | {"dummy_spikes"} <= set(ids)
    for t in ids:
        assert validate(load_spec(t), t) == [], t


def test_dummy_is_frozen_and_pilots_are_not():
    frozen(load_spec("dummy_spikes"))
    for t in PILOTS:
        spec = load_spec(t)
        assert spec["selection"] == "pilot_pre_rule"
        with pytest.raises(SpecError, match="unfilled"):
            frozen(spec)


def test_fills_reports_paths():
    assert fills({"a": {"b": ["x", "<<FILL: y>>"]}, "c": 1}) == ["a.b[1]"]


def test_cross_field_rules():
    spec = load_spec("dummy_spikes")
    bad = copy.deepcopy(spec)
    bad["data"]["truth"] = "{data_root}/tasks/dummy_spikes/test_labels.csv"
    assert any("truth_root" in e for e in validate(bad))
    bad = copy.deepcopy(spec)
    bad["baselines"]["automl"]["budget"]["wall_time_s"] = 1
    assert any("must equal budgets.wall_time_s" in e for e in validate(bad))
    bad = copy.deepcopy(spec)
    bad["budgets"]["wall_time_s"] = "ten minutes"
    assert validate(bad)
    bad = copy.deepcopy(spec)
    del bad["contamination"]
    assert validate(bad)
