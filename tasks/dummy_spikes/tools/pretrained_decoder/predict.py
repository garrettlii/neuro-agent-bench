"""Pretrained direction decoder for the dummy spike task.

Usage: python tools/pretrained_decoder/predict.py INPUT.npy OUTPUT.csv
INPUT.npy: integer spike counts, shape (n_trials, 40). OUTPUT.csv: header `row,label`,
one row per input trial, label in dir0..dir7.
"""
import csv
import sys
from pathlib import Path

import numpy as np

w = np.load(Path(__file__).with_name("weights.npz"))
x = np.load(sys.argv[1])
pred = np.argmax(x @ w["log_rates"].T - np.exp(w["log_rates"]).sum(1), axis=1)
with open(sys.argv[2], "w", newline="") as fh:
    out = csv.writer(fh)
    out.writerow(["row", "label"])
    out.writerows((i, w["labels"][k]) for i, k in enumerate(pred))
