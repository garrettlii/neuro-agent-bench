import numpy as np

from nabench.metrics import accuracy, read_predictions
from nabench.scoring import _tokens, ci, completeness, ratio


def test_normalized_score_anchors():
    s, o = np.array([0.1]), np.array([0.6])
    assert ratio(np.array([0.1]), s, o)[0] == 0 and ratio(np.array([0.6]), s, o)[0] == 1
    assert ratio(np.array([0.85]), s, o)[0] == 1.5
    lower_is_better = ratio(np.array([20.0]), np.array([40.0]), np.array([30.0]))[0]  # e.g. error rates
    assert lower_is_better == 2.0
    assert np.isnan(ci(np.array([np.nan]), float("nan"))[1])


def test_predictions_parsed_defensively(tmp_path):
    truth = {"a": "x", "b": "y", "c": "z"}
    f = tmp_path / "p.csv"
    f.write_text("case_id,prediction\na,x\nb,y\nb,q\nzz,x\n")
    preds, notes = read_predictions(f, truth)
    assert preds == {"a": "x"} and notes["duplicate"] == 1 and notes["unknown"] == 1
    assert list(accuracy(preds, truth)) == [1.0, 0.0, 0.0]
    f.write_text("id,label\na,x\n")
    assert read_predictions(f, truth)[0] == {}
    assert read_predictions(tmp_path / "missing.csv", truth)[0] == {}


def test_completeness_counts_missing_and_errors():
    manifest = {"runs": [{"run_id": "t__C1__m__r1"}, {"run_id": "t__C2__m__r1"}], "specs": {"t": {}}}
    subs = [{"run_id": r} for r in ("t__C1__m__r1", "t__baseline__simple__r1", "t__baseline__sota__r1", "t__baseline__automl__r1")]
    runs = [{"run_id": "t__C2__m__r1", "status": "error"}]
    st = completeness(manifest, subs, runs)
    assert st["missing"] == ["t__C2__m__r1"] and st["errors_pending"] == 1
    assert completeness(manifest, subs + [{"run_id": "t__C2__m__r1"}], runs)["missing"] == []


def test_tokens_in_include_cached_input():
    claude = {"input_tokens": 14, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000, "output_tokens": 5}
    codex = {"input_tokens": 1200, "cached_input_tokens": 1000, "output_tokens": 5, "reasoning_output_tokens": 2}
    assert _tokens([{"tokens": claude}]) == "1,114 / 5" and _tokens([{"tokens": codex}]) == "1,200 / 5"
