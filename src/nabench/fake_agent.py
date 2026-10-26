"""Deterministic stand-in for an agent CLI, for harness tests: no model call.

Run in a dummy_spikes workspace. Uses the pretrained decoder tool when the
workspace has it, otherwise a nearest-centroid decoder fitted on dev data, and
writes submission/predictions.csv. Prints zero token use in CLI JSON form.
Runs from source passed with -c, so it imports nothing from this package.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

x_dev = np.load("data/dev/spikes.npy")
with open("data/dev/labels.csv", newline="") as fh:
    y_dev = [row["label"] for row in csv.DictReader(fh)]
x_test = np.load("data/test/spikes.npy")
ids = Path("data/test/case_ids.csv").read_text().split()[1:]

tool = Path("tools/pretrained_decoder/predict.py")
if tool.exists():
    subprocess.run([sys.executable, str(tool), "data/test/spikes.npy", "submission/tool_out.csv"], check=True)
    with open("submission/tool_out.csv", newline="") as fh:
        pred = [row["label"] for row in csv.DictReader(fh)]
else:
    labels = sorted(set(y_dev))
    y = np.array([labels.index(v) for v in y_dev])
    centroids = np.stack([np.sqrt(x_dev[y == k]).mean(0) for k in range(len(labels))])
    dist = ((np.sqrt(x_test)[:, None, :] - centroids[None]) ** 2).sum(-1)
    pred = [labels[k] for k in dist.argmin(1)]

with open("submission/predictions.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["case_id", "prediction"])
    w.writerows(zip(ids, pred))
print(json.dumps({"result": "done", "usage": {"input_tokens": 0, "output_tokens": 0}}))
