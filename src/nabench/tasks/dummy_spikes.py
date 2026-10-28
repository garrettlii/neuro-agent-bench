"""Synthetic harness-test task: decode which of 8 movement directions was shown
from simulated spike counts of 40 cosine-tuned Poisson neurons.

`build` writes dev data, test inputs and a pretrained decoder (the task's
"trained-model tool", fitted on a separate simulated pretraining set) under
data_root, and test labels plus Bayes-optimal predictions (the noise ceiling)
under truth_root. Baselines and the scramble control read only the workspace.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..metrics import read_cases, write_cases

N_CLASSES, N_NEURONS = 8, 40
N_DEV_PER_CLASS, N_TEST, N_PRETRAIN = 20, 400, 8000
LABELS = [f"dir{k}" for k in range(N_CLASSES)]


def _rates(rng: np.random.Generator) -> np.ndarray:
    theta = 2 * np.pi * np.arange(N_CLASSES) / N_CLASSES
    pref = rng.uniform(0, 2 * np.pi, N_NEURONS)
    base = rng.uniform(0.5, 2.0, N_NEURONS)
    return base * np.exp(0.6 * np.cos(theta[:, None] - pref[None, :]))  # (classes, neurons)


def _sample(rates, y, rng):
    return rng.poisson(rates[y]).astype(np.int64)


def poisson_nb_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-class mean counts with a small prior (log rates, classes x neurons)."""
    return np.log(np.stack([(x[y == k].sum(0) + 0.5) / ((y == k).sum() + 1) for k in range(N_CLASSES)]))


def poisson_nb_predict(log_rates: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.argmax(x @ log_rates.T - np.exp(log_rates).sum(1), axis=1)


def build(data_dir: Path, truth_dir: Path, seed: int = 20261026) -> None:
    rng = np.random.default_rng(seed)
    rates = _rates(rng)
    y_dev = np.repeat(np.arange(N_CLASSES), N_DEV_PER_CLASS)
    rng.shuffle(y_dev)
    y_test = rng.integers(0, N_CLASSES, N_TEST)
    y_pre = rng.integers(0, N_CLASSES, N_PRETRAIN)
    x_dev, x_test, x_pre = (_sample(rates, y, rng) for y in (y_dev, y_test, y_pre))
    for d in ("dev", "test", "tools/pretrained_decoder"):
        (data_dir / d).mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.chmod(0o700)
    np.save(data_dir / "dev" / "spikes.npy", x_dev)
    write_cases(data_dir / "dev" / "labels.csv", {f"dev{i:04d}": LABELS[k] for i, k in enumerate(y_dev)}, "label")
    np.save(data_dir / "test" / "spikes.npy", x_test)
    test_ids = [f"test{i:04d}" for i in range(N_TEST)]
    (data_dir / "test" / "case_ids.csv").write_text("case_id\n" + "\n".join(test_ids) + "\n")
    np.savez(data_dir / "tools" / "pretrained_decoder" / "weights.npz",
             log_rates=poisson_nb_fit(x_pre, y_pre), labels=np.array(LABELS))
    write_cases(truth_dir / "test_labels.csv", dict(zip(test_ids, (LABELS[k] for k in y_test))), "label")
    oracle = poisson_nb_predict(np.log(rates), x_test)
    write_cases(truth_dir / "oracle_predictions.csv", dict(zip(test_ids, (LABELS[k] for k in oracle))))


def _load(workspace: Path):
    d = workspace / "data"
    dev = read_cases(d / "dev" / "labels.csv", "label")
    y = np.array([LABELS.index(v) for v in dev.values()])
    ids = (d / "test" / "case_ids.csv").read_text().split()[1:]
    return np.load(d / "dev" / "spikes.npy"), y, np.load(d / "test" / "spikes.npy"), ids


def simple_baseline(workspace: Path, budget: dict) -> dict[str, str]:
    """Most frequent dev label for every test case (ties: first label)."""
    _, y, _, ids = _load(workspace)
    top = LABELS[int(np.bincount(y, minlength=N_CLASSES).argmax())]
    return {cid: top for cid in ids}


def sota_baseline(workspace: Path, budget: dict) -> dict[str, str]:
    """Poisson naive Bayes fitted on dev (the correct model family for this simulation)."""
    x, y, xt, ids = _load(workspace)
    return {cid: LABELS[k] for cid, k in zip(ids, poisson_nb_predict(poisson_nb_fit(x, y), xt))}


def automl_baseline(workspace: Path, budget: dict) -> dict[str, str]:
    """Model and hyperparameter search by 5-fold CV on dev within the agent's wall-time budget."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier

    x, y, xt, ids = _load(workspace)
    x, xt = np.sqrt(x), np.sqrt(xt)
    candidates = [LogisticRegression(C=c, max_iter=2000) for c in (0.01, 0.1, 1.0, 10.0)]
    candidates += [KNeighborsClassifier(k) for k in (5, 15)] + [GaussianNB(), RandomForestClassifier(300, random_state=0)]
    deadline = time.time() + float(budget["wall_time_s"])
    best, best_score = None, -np.inf
    for model in candidates:
        if time.time() > deadline:
            break
        score = cross_val_score(model, x, y, cv=5).mean()
        if score > best_score:
            best, best_score = model, score
    best.fit(x, y)
    return {cid: LABELS[k] for cid, k in zip(ids, best.predict(xt))}


def scramble(data_dir: Path, rng: np.random.Generator, spec: dict) -> None:
    """Permute dev labels across dev trials and test inputs across test cases (`spec` unused)."""
    labels = read_cases(data_dir / "dev" / "labels.csv", "label")
    write_cases(data_dir / "dev" / "labels.csv", dict(zip(labels, rng.permutation(list(labels.values())))), "label")
    x = np.load(data_dir / "test" / "spikes.npy")
    np.save(data_dir / "test" / "spikes.npy", x[rng.permutation(len(x))])


def noise_ceiling(spec: dict) -> dict[str, str]:
    """Bayes-optimal predictions from the generating rates, written by `build`."""
    return read_cases(Path(spec["data"]["truth"]).parent / "oracle_predictions.csv", "prediction")
