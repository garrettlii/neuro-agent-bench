"""T15 intracortical speech (Brain-to-Text '25): metric only. The data build and
baselines are not written yet.

Scored as the competition scores it: aggregate word error rate (summed word edit
distance over summed reference words) after the organizers' text normalization.
"""
from __future__ import annotations

import re

import numpy as np


def normalize(sentence: str) -> str:
    """`remove_punctuation` from Neuroprosthetics-Lab/nejm-brain-to-text,
    model_training/evaluate_model_helpers.py @ e93cff1e2e12323cef2665f24eb1b35adb6eaa92,
    which evaluate_model.py applies to both the true and the predicted sentence before WER."""
    sentence = re.sub(r"[^a-zA-Z\- ']", "", sentence)
    sentence = sentence.replace("- ", " ").lower()
    sentence = sentence.replace("--", "")
    sentence = sentence.replace(" '", "'")
    return " ".join(sentence.split())


def word_edits(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance over words (substitutions, insertions, deletions)."""
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def wer_share(pred: dict[str, str], truth: dict[str, str]) -> np.ndarray:
    """Per case: word edits x N / total reference words, so the mean over cases is the aggregate
    WER. A missing prediction scores as deletion of every reference word; in truth order."""
    refs = [normalize(t).split() for t in truth.values()]
    edits = [word_edits(r, normalize(pred.get(cid, "")).split()) for cid, r in zip(truth, refs)]
    return np.array(edits, dtype=float) * (len(refs) / sum(map(len, refs)))
