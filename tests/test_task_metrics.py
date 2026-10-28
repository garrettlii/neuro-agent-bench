"""Task metrics defined by their benchmarks' scoring contracts."""
import re
from functools import lru_cache

import numpy as np
import pytest

from nabench.tasks.bwb_neuropixels_wheel import session_r2
from nabench.tasks.t15_speech import normalize, wer_share, word_edits


def remove_punctuation(sentence):
    # verbatim from Neuroprosthetics-Lab/nejm-brain-to-text model_training/evaluate_model_helpers.py @ e93cff1e
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()

    sentence = sentence.strip()
    sentence = ' '.join([word for word in sentence.split() if word != ''])

    return sentence


def edits_reference(a: tuple, b: tuple) -> int:
    """Independent recursive Levenshtein over words."""
    @lru_cache(maxsize=None)
    def d(i, j):
        if i == 0 or j == 0:
            return i + j
        return min(d(i - 1, j) + 1, d(i, j - 1) + 1, d(i - 1, j - 1) + (a[i - 1] != b[j - 1]))
    return d(len(a), len(b))


@pytest.mark.parametrize("s", ["Hello, World!", "I don't know -- maybe.", "well- that's it", "go 'cause I said so",
                               "  Spaces   everywhere  ", "Numbers 42 and 7th", "A-B test - ok", ""])
def test_normalize_matches_organizers_code(s):
    assert normalize(s) == remove_punctuation(s)


def test_word_edits():
    assert word_edits("the cat sat".split(), "the cat sat".split()) == 0
    assert word_edits("the cat sat".split(), "a cat sat down".split()) == 2  # one substitution, one insertion
    assert word_edits("the cat sat".split(), []) == 3
    assert word_edits([], "x y".split()) == 2


def test_wer_share_mean_is_aggregate_wer():
    truth = {"c1": "The cat sat on the mat.", "c2": "I don't know.", "c3": "Where is it?", "c4": "Go home"}
    pred = {"c1": "the cat sat on a mat", "c2": "i do not know", "c3": "WHERE IS IT", "x9": "unknown case"}  # c4 missing
    refs = [tuple(remove_punctuation(t).split()) for t in truth.values()]
    hyps = [tuple(remove_punctuation(pred.get(c, "")).split()) for c in truth]
    aggregate = sum(edits_reference(r, h) for r, h in zip(refs, hyps)) / sum(map(len, refs))
    scores = wer_share(pred, truth)
    assert scores.mean() == pytest.approx(aggregate)
    assert aggregate == pytest.approx((1 + 2 + 0 + 2) / (6 + 3 + 3 + 2))
    assert scores[3] == pytest.approx(2 * 4 / 14)  # missing: both reference words deleted


def test_session_r2_hand_computed():
    truth = {"s1": "1 2 3 4", "s2": "0 1", "s3": "0 0 2 2"}
    # s1: SS_res = 1, mean 2.5, SS_tot = 5 -> 0.8; s2 worse than the mean -> clipped to 0;
    # s3 pools both trials' frames: SS_res = 2, SS_tot = 4 -> 0.5
    pred = {"s1": "1 2 3 5", "s2": "5 5", "s3": "0 0 1 1"}
    np.testing.assert_allclose(session_r2(pred, truth), [0.8, 0.0, 0.5])
    assert session_r2({"s1": "1 2 3 4"}, {"s1": "1 2 3 4"})[0] == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [None, "", "1 2 3", "1 2 3 4 5", "1 2 x 4", "1 2 nan 4", "1 2 inf 4"])
def test_session_r2_bad_traces_score_zero(bad):
    pred = {} if bad is None else {"s1": bad}
    assert session_r2(pred, {"s1": "1 2 3 4"})[0] == 0.0
