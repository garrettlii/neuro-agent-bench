"""BrainWideBench TS1 wheel speed: metric, data build, C3 scramble and baselines.

One case per evaluation session. Its label and prediction are the session's wheel-speed
trace over every test frame, as space-separated numbers in a fixed frame order.

`build` reads the benchmark's processed evaluation sessions (all_units build, one .h5 per
session) and serves what its TS1 loader serves for wheel_speed at the default context
(ibl-bwb @ 6fbd666f, src/ts1/ts1_dataset.py and src/ts1/ts1_test_mixin.py): one sample per
trial window [movement onset, movement onset + 1 s), holding the spikes of every unit in the
window and the 50 frames of the 50 Hz wheel-speed series inside it. Layouts: docs/DATA.md.

The baselines reproduce the benchmark's Linear and POYO+ (ibl-bwb @ 6fbd666f) and add an AutoML
anchor; they run in the task's own environment (see _child).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..metrics import write_cases

FS = 50  # Hz; ibl_bwb_eval/tasks/ts1.py BEHAVIOR_SFREQ
N_FRAMES = 50  # BEHAVIOR_SFREQ * TARGET_WINDOW (1.0 s)
BUILD, VERSION = "all_units", "0.1.0"  # the build and derived_version TS1 reports on


def _trace(s: str) -> np.ndarray:
    return np.array(s.split(), dtype=float)


def session_r2(pred: dict[str, str], truth: dict[str, str]) -> np.ndarray:
    """R^2 over all test frames of each session, clipped at 0 (arXiv 2609.22064v1 Appendix E.3;
    ibl_bwb_eval 0.1.0 pools a session's frames in scoring/ts1.py and clips in scoring/aggregation.py).
    A missing, unparseable, non-finite or wrong-length trace scores 0; in truth order."""
    out = []
    for cid, label in truth.items():
        y = _trace(label)
        try:
            yhat = _trace(pred[cid])
        except (KeyError, ValueError):
            out.append(0.0)
            continue
        if yhat.shape != y.shape or not np.isfinite(yhat).all():
            out.append(0.0)
            continue
        r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out.append(max(float(r2), 0.0))
    return np.array(out)


def in_domain(windows: np.ndarray, domain: np.ndarray) -> np.ndarray:
    """Which (start, end) rows of `windows` lie inside one of the sorted, disjoint `domain`
    intervals. The benchmark intersects its windows with each domain, which would clip a
    window straddling a domain edge into a shorter sample; that raises here instead."""
    a, b = windows[:, 0], windows[:, 1]
    j = np.searchsorted(domain[:, 0], a, side="right") - 1  # last interval starting at or before a
    inside = (j >= 0) & (b <= domain[np.maximum(j, 0), 1])
    k = np.minimum(np.searchsorted(domain[:, 1], a, side="right"), len(domain) - 1)  # first ending after a
    overlaps = (domain[k, 1] > a) & (domain[k, 0] < b)
    if np.any(overlaps & ~inside):
        raise ValueError(f"{int((overlaps & ~inside).sum())} windows straddle a domain edge")
    return inside


def window_spikes(times: np.ndarray, units: np.ndarray, windows: np.ndarray):
    """Spikes in [start, end) of each window with times relative to its start, as
    torch_brain's IrregularTimeSeries.slice returns them. Trial i is ptr[i]:ptr[i + 1]."""
    lo, hi = np.searchsorted(times, windows[:, 0]), np.searchsorted(times, windows[:, 1])
    idx = np.concatenate([np.arange(l, h) for l, h in zip(lo, hi)] + [np.zeros(0, int)])
    n = hi - lo
    rel = (times[idx] - np.repeat(windows[:, 0], n)).astype(np.float32)
    # float32 rounds a spike within ~3e-8 s of the window end up onto it; keep times in [0, length)
    length = np.repeat(windows[:, 1] - windows[:, 0], n).astype(np.float32)
    rel = np.minimum(rel, np.nextafter(length, np.float32(0)))
    return rel, units[idx].astype(np.int32), np.concatenate([[0], np.cumsum(n)])


def window_frames(values: np.ndarray, t0: float, windows: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """The samples of a FS series starting at t0 inside each [start, end) window, picked as
    torch_brain's RegularTimeSeries.slice picks them: both edges snapped up to the grid."""
    def idx(t):
        x = (t - t0) * FS
        r = np.round(x)
        return np.ceil(np.where(np.abs(x - r) < eps, r, x)).astype(int)

    i0, i1 = idx(windows[:, 0]), idx(windows[:, 1])
    if np.any(i1 - i0 != N_FRAMES):
        raise ValueError(f"{int((i1 - i0 != N_FRAMES).sum())} windows do not hold {N_FRAMES} frames")
    return values[i0[:, None] + np.arange(N_FRAMES)].astype(np.float32)


def session_samples(rec: dict) -> dict[str, dict]:
    """Per split (train, val, test): the benchmark's wheel_speed sampling intervals in time
    order (ts1_dataset.py get_sampling_intervals: movement windows inside the task-aligned
    domain, the split's domain and the wheel series' domain) and their spikes and frames.
    Trials in the 20 s gaps between splits belong to no split."""
    w = rec["windows"]
    base = in_domain(w, rec["task_domain"]) & in_domain(w, rec["wheel_domain"])
    out = {}
    for split in ("train", "val", "test"):
        keep = base & in_domain(w, rec[f"{split}_domain"])
        t, u, ptr = window_spikes(rec["times"], rec["units"], w[keep])
        out[split] = {"trial_index": rec["trial_index"][keep], "spike_times": t, "spike_units": u, "spike_ptr": ptr,
                      "wheel_speed": window_frames(rec["wheel"], rec["wheel_t0"], w[keep]), "windows": w[keep]}
    return out


def _intervals(group) -> np.ndarray:
    return np.stack([group["start"][:], group["end"][:]], axis=1)


def _read(path: Path) -> dict:
    """The arrays `session_samples` needs from one session's .h5."""
    import h5py

    with h5py.File(path, "r") as f:
        meta = {k: (v.decode() if isinstance(v, bytes) else str(v)) for k, v in f["brainset"].attrs.items()}
        if (meta.get("unit_filtering"), meta.get("derived_version")) != (BUILD, VERSION):
            raise ValueError(f"{path.name}: build {meta.get('unit_filtering')} {meta.get('derived_version')}")
        wheel, mw = f["wheel"], f["task_aligned_intervals/movement_window"]
        if wheel.attrs["sampling_rate"] != FS:
            raise ValueError(f"{path.name}: wheel at {wheel.attrs['sampling_rate']} Hz")
        rec = {
            "times": f["spikes/timestamps"][:], "units": f["spikes/unit_index"][:], "n_units": len(f["units/id"]),
            "span": np.array([f["spikes/domain/start"][0], f["spikes/domain/end"][-1]]),
            "windows": _intervals(mw), "trial_index": mw["trial_index"][:],
            "task_domain": _intervals(f["task_aligned_intervals/domain"]),
            "wheel": wheel["wheel_speed"][:], "wheel_t0": float(wheel["domain/start"][0]),
            "wheel_domain": _intervals(wheel["domain"]),
            **{f"{s}_domain": _intervals(f[f"ts1_{s}_domain"]) for s in ("train", "val", "test")},
        }
    if np.any(np.diff(rec["times"]) < 0):
        raise ValueError(f"{path.name}: spikes are not sorted")
    return rec


def _arrays(cid: str, s: dict, n_units: int) -> dict[str, np.ndarray]:
    keys = ("trial_index", "spike_times", "spike_units", "spike_ptr")
    return {f"{cid}_n_units": np.array(n_units), **{f"{cid}_{k}": s[k] for k in keys}}


def build(raw_dir: Path, data_dir: Path, truth_dir: Path, seed: int = 20261027) -> None:
    """Every session .h5 in raw_dir becomes one case. Case ids are assigned in a seeded random
    order so they carry no session identity; the mapping stays in truth_dir/case_map.csv. Trials
    are renumbered 0, 1, ... in time order per session, dev then test as one sequence, so the
    numbers do not reveal excluded trials. data_dir/scramble/<case>.npz keeps what `scramble`
    needs (the full spike train and the trial windows); it is never staged."""
    paths = sorted(Path(raw_dir).glob("*.h5"))
    ids = [f"s{i + 1:02d}" for i in range(len(paths))]
    order = np.random.default_rng(seed).permutation(len(paths))
    for d in ("dev", "test", "scramble"):
        (data_dir / d).mkdir(parents=True, exist_ok=True)
    dev, dev_y, test, labels, test_rows = {}, {}, {}, {}, []
    for cid, k in zip(ids, order):
        rec = _read(paths[k])
        s = session_samples(rec)
        tr, va, te = s["train"], s["val"], s["test"]
        n_dev = len(tr["trial_index"]) + len(va["trial_index"])
        merged = {"trial_index": np.arange(n_dev),
                  "spike_times": np.concatenate([tr["spike_times"], va["spike_times"]]),
                  "spike_units": np.concatenate([tr["spike_units"], va["spike_units"]]),
                  "spike_ptr": np.concatenate([tr["spike_ptr"], va["spike_ptr"][1:] + tr["spike_ptr"][-1]])}
        dev |= _arrays(cid, merged, rec["n_units"])
        dev[f"{cid}_is_val"] = np.repeat([False, True], [len(tr["trial_index"]), len(va["trial_index"])])
        dev_y[cid] = np.concatenate([tr["wheel_speed"], va["wheel_speed"]])
        te = {**te, "trial_index": np.arange(n_dev, n_dev + len(te["trial_index"]))}
        test |= _arrays(cid, te, rec["n_units"])
        labels[cid] = " ".join(f"{v:.9g}" for v in te["wheel_speed"].ravel())
        test_rows += [f"{cid},{t}" for t in te["trial_index"]]
        np.savez(data_dir / "scramble" / f"{cid}.npz", times=rec["times"], units=rec["units"].astype(np.int32),
                 span=rec["span"], dev_windows=np.concatenate([tr["windows"], va["windows"]]), test_windows=te["windows"],
                 source_trial_index=np.concatenate([tr["trial_index"], va["trial_index"], s["test"]["trial_index"]]))
        del rec, s
    truth_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.chmod(0o700)
    np.savez(data_dir / "dev" / "sessions.npz", **dev)
    np.savez(data_dir / "dev" / "wheel_speed.npz", **dev_y)
    np.savez(data_dir / "test" / "sessions.npz", **test)
    (data_dir / "test" / "trial_index.csv").write_text("case_id,trial_index\n" + "\n".join(test_rows) + "\n")
    (data_dir / "test" / "case_ids.csv").write_text("case_id\n" + "\n".join(ids) + "\n")
    write_cases(truth_dir / "test_labels.csv", labels, "label")
    write_cases(truth_dir / "case_map.csv", {cid: paths[k].stem for cid, k in zip(ids, order)}, "session")


def circular_shift(times: np.ndarray, units: np.ndarray, span: np.ndarray, offset: float):
    """A sorted spike train moved later by `offset` inside span = (start, end), the spikes pushed
    past the end wrapping round to the start; still sorted, with no sort needed."""
    start, end = span
    k = np.searchsorted(times, end - offset)
    return (np.concatenate([times[k:] + (offset - (end - start)), times[:k] + offset]),
            np.concatenate([units[k:], units[:k]]))


def scramble(data_dir: Path, rng: np.random.Generator, spec: dict) -> None:
    """C3: each session's whole spike train circularly shifted against behaviour and trial events
    by one offset drawn uniformly from 10% to 90% of the session's duration, then every dev and
    test window re-cut from it; trials, labels and unit counts stay as they are. Reads the
    harness-only data_dir/scramble/<case>.npz that `build` writes next to the staged files."""
    src = Path(spec["data"]["dev_inputs"][0]).parents[1] / "scramble"
    ids = (data_dir / "test" / "case_ids.csv").read_text().split()[1:]
    arrays = {d: dict(np.load(data_dir / d / "sessions.npz")) for d in ("dev", "test")}
    for cid in ids:
        with np.load(src / f"{cid}.npz") as z:
            span = z["span"]
            t, u = circular_shift(z["times"], z["units"], span, rng.uniform(0.1, 0.9) * (span[1] - span[0]))
            for d in ("dev", "test"):
                a, w = arrays[d], z[f"{d}_windows"]
                if len(w) != len(a[f"{cid}_trial_index"]):
                    raise ValueError(f"{cid}: scramble source does not match the staged {d} trials")
                a[f"{cid}_spike_times"], a[f"{cid}_spike_units"], a[f"{cid}_spike_ptr"] = window_spikes(t, u, w)
    for d, a in arrays.items():
        np.savez(data_dir / d / "sessions.npz", **a)


# Baselines. They need torch, and POYO+ also torch_brain and the ibl-bwb source, none of which the
# harness environment has, so each runs in a child process of the task's environment
# (data_root/envs/bwb): `python -I -m nabench.tasks.bwb_neuropixels_wheel <kind> <workspace> <budget json> <out.csv>`.
# The train / val / predict split arguments let the same code run the validation-only checks.

EVAL_SEEDS = (43, 44, 45, 46, 47)  # ibl_bwb_eval/protocol.py EVAL_SEEDS
LINEAR_SWEEP = {"trials": 100, "seed": 0}  # ts1/configs/tune.yaml ray.num_samples, ray.search_seed
POYO_LRS, POYO_SWEEP_SEED = (1e-5, 2e-5, 5e-5, 1e-4, 5e-4), 42  # poyo_plus_gradual_unfreezing.yaml; protocol.py SWEEP_SEED
# "published": the hyperparameters the benchmark selected on each session's val trials, read from the harness-only
# data_root/tasks/bwb_neuropixels_wheel/reference/ (W&B ibl-bwb/ts1-linear-ss, ts1-poyo_plus-ft-seeds);
# "sweep": rerun the benchmark's selection here (Linear: 100 Optuna trials per session; POYO+: 5 learning rates).
SELECTION = "published"
SOTA_SEEDS = 5  # D-010: the 5-seed ensemble, run on a GPU machine (scripts/sota_gpu.py)
SOTA_INPUTS = ("dev/sessions.npz", "dev/wheel_speed.npz", "test/sessions.npz", "test/case_ids.csv")  # what load_cases reads


def load_cases(workspace: Path) -> list[dict]:
    """The staged data per case: trials as (spike_times, spike_units) pairs, dev labels and is_val."""
    d = Path(workspace) / "data"
    dev, test, y = (np.load(d / f) for f in ("dev/sessions.npz", "test/sessions.npz", "dev/wheel_speed.npz"))

    def trials(z, cid):
        p, t, u = z[f"{cid}_spike_ptr"], z[f"{cid}_spike_times"], z[f"{cid}_spike_units"]
        return [(t[a:b], u[a:b]) for a, b in zip(p[:-1], p[1:])]

    return [{"id": cid, "n_units": int(dev[f"{cid}_n_units"]), "dev": trials(dev, cid), "is_val": dev[f"{cid}_is_val"],
             "y": y[cid], "test": trials(test, cid)} for cid in (d / "test" / "case_ids.csv").read_text().split()[1:]]


def pooled_r2(pred: np.ndarray, y: np.ndarray) -> float:
    """R^2 over all frames of a set of trials, as torchmetrics R2Score on (trials * 50, 1)."""
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _pick(trials, mask):
    return [t for t, m in zip(trials, mask) if m]


def _torch_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def _spike_bins(trials, n_units: int, bin_size: float):
    """Flat (bin * n_units + unit) index of every spike, trial by trial, and each trial's first
    spike: torch_brain's bin_spikes on [0, 1) (bin = floor(t / bin_size)) flattened as the
    benchmark's Linear flattens its (T, N) counts."""
    n_bins = round(1.0 / bin_size)
    t = np.concatenate([x[0] for x in trials]).astype(np.float64)
    u = np.concatenate([x[1] for x in trials]).astype(np.int64)
    b = np.minimum(np.floor(t * (1.0 / bin_size)).astype(np.int64), n_bins - 1)
    return b * n_units + u, np.cumsum([0] + [len(x[0]) for x in trials[:-1]]), n_bins * n_units


def spike_counts(trials, n_units: int, bin_size: float) -> np.ndarray:
    """Each trial's (bins x units) spike counts flattened, (n_trials, n_bins * n_units) float32."""
    idx, off, n_in = _spike_bins(trials, n_units, bin_size)
    row = np.repeat(np.arange(len(trials)), np.diff(np.append(off, len(idx))))
    return np.bincount(row * n_in + idx, minlength=len(trials) * n_in).reshape(len(trials), n_in).astype(np.float32)


def linear_fit(train, y_train, val, y_val, targets, n_units: int, *, bin_size=0.02, num_epochs=100, batch_size=32,
               weight_decay=1e-4, base_lr=1e-3, pct_start=0.5, div_factor=1.0, seed=42, device="cpu", dense_limit=2.5e8):
    """The benchmark's Linear baseline (ts1/models/single_session/linear) trained as TS1EvalTrainer
    trains it (ts1/ts1_eval_trainer.py, core/trainer.py, ts1/configs/train.yaml; defaults here):
    one linear layer from the flattened (bins x units) spike counts to the 50 frames, nn.Linear
    initialisation, MSE, AdamW (weight decay on the weight only) under OneCycleLR, gradient norm
    clipped at 1, shuffled batches with the last partial one dropped, val R^2 after every epoch,
    early stopping (patience 20 from epoch 100) and the best-val weights kept. When the counts
    would not fit in memory the layer runs as an embedding bag over spikes (the same product).
    Returns (best val R^2, predictions for `targets`); raises ValueError if a batch cannot be filled."""
    import torch
    import torch.nn.functional as F

    if len(train) < batch_size:
        raise ValueError(f"batch size {batch_size} > {len(train)} train trials")
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dev = torch.device(device)
    n_in = round(1.0 / bin_size) * n_units
    dense = (len(train) + len(val)) * n_in <= dense_limit  # counts as a dense float32 matrix, 1 GB by default

    def features(trials):
        if dense:
            return torch.as_tensor(spike_counts(trials, n_units, bin_size), device=dev)
        i, o, _ = _spike_bins(trials, n_units, bin_size)
        return torch.as_tensor(i, device=dev), np.append(o, len(i))

    def forward(f, rows=None):
        if dense:
            return (f if rows is None else f[rows]) @ w + b
        i, o = f
        rows = range(len(o) - 1) if rows is None else rows.tolist()
        parts = [i[o[r]:o[r + 1]] for r in rows]
        return F.embedding_bag(torch.cat(parts), w, torch.as_tensor(np.cumsum([0] + [len(p) for p in parts[:-1]]), device=dev),
                               mode="sum") + b

    def predict(f):
        with torch.no_grad():
            return forward(f).cpu().numpy()

    x_train, x_val = features(train), features(val)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=dev)
    bound = 1 / np.sqrt(n_in)
    w = torch.nn.Parameter(torch.empty(n_in, y_train.shape[1], device=dev).uniform_(-bound, bound))
    b = torch.nn.Parameter(torch.empty(y_train.shape[1], device=dev).uniform_(-bound, bound))
    steps = len(train) // batch_size
    opt = torch.optim.AdamW([{"params": [w], "weight_decay": weight_decay}, {"params": [b], "weight_decay": 0.0}],
                            lr=base_lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=base_lr, epochs=num_epochs, steps_per_epoch=steps,
                                                pct_start=pct_start, anneal_strategy="cos", div_factor=div_factor,
                                                final_div_factor=1e4)
    best, best_state, patience = -np.inf, None, 20
    for epoch in range(num_epochs):
        perm = torch.randperm(len(train), generator=gen)
        for s in range(steps):
            rows = perm[s * batch_size:(s + 1) * batch_size]
            loss = F.mse_loss(forward(x_train, rows), yt[rows.to(dev)])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([w, b], 1.0)
            opt.step()
            sched.step()
        score = pooled_r2(predict(x_val), y_val)
        if np.isfinite(score) and score > best:
            best, best_state, patience = score, (w.detach().clone(), b.detach().clone()), 20
        elif epoch >= 100:
            patience -= 1
        if patience <= 0:
            break
    if best_state is not None:
        with torch.no_grad():
            w.copy_(best_state[0])
            b.copy_(best_state[1])
    if not targets:
        return best, np.zeros((0, y_train.shape[1]), np.float32)
    del x_train, x_val
    return best, predict(features(targets))


def linear_sweep(train, y_train, val, y_val, n_units: int, trials: int = LINEAR_SWEEP["trials"],
                 seed: int = LINEAR_SWEEP["seed"], device="cpu") -> dict:
    """The benchmark's Linear hyperparameter search (core/tune.py, ts1/configs/tune.yaml,
    Linear.create_search_space): Optuna TPE seeded with search_seed 0 over bin size, epochs, batch
    size, weight decay, learning rate and the OneCycle shape, each trial trained with seed 0 and
    scored by its best val R^2; trials whose batch cannot be filled are pruned. Ray ran 4 trials at
    a time; this runs them one after another, so the sampled sequence differs."""
    import optuna

    def objective(trial):
        params = {"bin_size": trial.suggest_categorical("model.bin_size", [0.001, 0.005, 0.01, 0.02, 0.04]),
                  "num_epochs": trial.suggest_int("num_epochs", 100, 500, step=200),
                  "batch_size": 2 ** trial.suggest_int("batch_size_log2", 4, 6, step=1),
                  "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
                  "base_lr": trial.suggest_float("base_lr", 1e-5, 1e-2, log=True),
                  "pct_start": trial.suggest_float("pct_start", 0.1, 0.9),
                  "div_factor": trial.suggest_float("div_factor", 1, 5, log=True)}
        if len(train) < params["batch_size"]:
            raise optuna.TrialPruned()
        trial.set_user_attr("params", params)
        return linear_fit(train, y_train, val, y_val, [], n_units, **params, seed=seed, device=device)[0]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    return study.best_trial.user_attrs["params"]


def linear_case(c: dict, targets, trials: int = LINEAR_SWEEP["trials"], device="cpu"):
    """Sweep on train / val, then the best configuration at every eval seed; per seed the best val
    R^2 and the predictions for `targets`."""
    tr, va = ~c["is_val"], c["is_val"]
    args = (_pick(c["dev"], tr), c["y"][tr], _pick(c["dev"], va), c["y"][va])
    params = linear_sweep(*args, c["n_units"], trials=trials, device=device)
    return params, [linear_fit(*args, targets, c["n_units"], **params, seed=s, device=device) for s in EVAL_SEEDS]


AUTOML_SEED = 0
AUTOML_SELECTION_SHARE = 0.875  # D-013: about 3.5 h of the 4 h budget for the search, the rest for the refits


def automl_candidates():
    """(name, bin size in s, estimator factory): a fixed grid, roughly cheapest first, then an endless
    seeded random search over wider ranges of the same families plus kernel ridge and PLS. Standard
    scikit-learn regressors on per-unit spike counts, square-root transformed, in equal bins over the
    1 s window; every model predicts the 50 frames jointly. Names are unique (automl_predict finds the
    selected candidate by name)."""
    import itertools

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = [(f"ridge a={a:g} bin={b:g}", b, lambda a=a: make_pipeline(StandardScaler(), Ridge(alpha=a)))
           for b in (0.1, 0.05, 0.2) for a in (1e1, 1e2, 1e3, 1e4, 1e5)]
    pca = lambda k=64: (StandardScaler(), PCA(k, random_state=0))  # noqa: E731
    out += [(f"pca64+knn k={k}", 0.1, lambda k=k: make_pipeline(*pca(), KNeighborsRegressor(k))) for k in (5, 20)]
    out += [("pca64+random forest", 0.1, lambda: make_pipeline(*pca(), RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0))),
            ("pca64+extra trees", 0.1, lambda: make_pipeline(*pca(), ExtraTreesRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0))),
            ("pca64+mlp", 0.1, lambda: make_pipeline(*pca(), StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 256), alpha=1e-3, max_iter=500,
                                                                                            early_stopping=True, random_state=0)))]
    yield from out

    rng = np.random.default_rng(AUTOML_SEED)
    log_u = lambda lo, hi: float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))  # noqa: E731
    for i in itertools.count(1):
        b, family = float(rng.choice([0.02, 0.05, 0.1, 0.2])), str(rng.choice(
            ["ridge", "pls", "kernel ridge", "knn", "random forest", "extra trees", "mlp"]))
        k = 0 if family in ("ridge", "pls") else int(rng.choice([16, 32, 64]))  # PCA first for the rest, so no fit runs on 150k features
        if family == "ridge":
            p = {"alpha": log_u(1e-1, 1e6)}
            est = lambda p=p: Ridge(**p)  # noqa: E731
        elif family == "pls":
            p = {"n_components": int(rng.integers(2, 33))}
            est = lambda p=p: PLSRegression(**p)  # noqa: E731
        elif family == "kernel ridge":
            p = {"alpha": log_u(1e-3, 1e2), "gamma": log_u(1e-2, 1e1) / k}
            est = lambda p=p: KernelRidge(kernel="rbf", **p)  # noqa: E731
        elif family == "knn":
            p = {"n_neighbors": int(rng.integers(2, 51)), "weights": str(rng.choice(["uniform", "distance"]))}
            est = lambda p=p: KNeighborsRegressor(**p)  # noqa: E731
        elif family in ("random forest", "extra trees"):
            p = {"n_estimators": int(rng.choice([100, 200, 400])), "min_samples_leaf": int(rng.integers(1, 11)),
                 "max_features": ["sqrt", 0.3, 1.0][int(rng.integers(3))]}
            est = lambda p=p, f=family: (RandomForestRegressor if f == "random forest" else ExtraTreesRegressor)(  # noqa: E731
                **p, n_jobs=-1, random_state=0)
        else:
            p = {"hidden_layer_sizes": [(64,), (128,), (256,), (128, 128), (256, 256)][int(rng.integers(5))],
                 "alpha": log_u(1e-5, 1e0), "learning_rate_init": log_u(1e-4, 1e-2)}
            est = lambda p=p: make_pipeline(StandardScaler(), MLPRegressor(**p, max_iter=500, early_stopping=True, random_state=0))  # noqa: E731
        desc = " ".join(f"{n}={v:.3g}" if isinstance(v, float) else f"{n}={v}" for n, v in p.items())
        yield (f"#{i} {family} {desc} pca={k} bin={b:g}", b,
               lambda k=k, est=est: make_pipeline(*(pca(k) if k else (StandardScaler(),)), est()))


def binned_counts(trials, n_units: int, bin_size: float) -> np.ndarray:
    """sqrt of each unit's spike count per bin, (n_trials, n_bins * n_units)."""
    return np.sqrt(spike_counts(trials, n_units, bin_size))


def automl_select(cases: list[dict], wall_time_s: float, searched: list | None = None) -> dict[str, tuple[str, float]]:
    """Fit every candidate on each case's train trials and score it on its val trials, candidate by
    candidate over all cases until the wall-time budget is spent; per case the best (name, val R^2).
    A fit that fails numerically counts as not chosen. `searched` gets (name, cases fitted, cases failed)
    per candidate tried."""
    import time

    deadline, best = time.time() + wall_time_s, {c["id"]: ("", -np.inf) for c in cases}
    for name, bin_size, make in automl_candidates():
        if time.time() > deadline:
            return best
        fitted = failed = 0
        for c in cases:
            if time.time() > deadline:
                break
            tr, va = ~c["is_val"], c["is_val"]
            x = binned_counts(c["dev"], c["n_units"], bin_size)
            try:
                score = pooled_r2(make().fit(x[tr], c["y"][tr]).predict(x[va]), c["y"][va])
            except (ValueError, np.linalg.LinAlgError):
                score, failed = -np.inf, failed + 1
            fitted += 1
            if score > best[c["id"]][1]:
                best[c["id"]] = (name, score)
        if searched is not None:
            searched.append((name, fitted, failed))
    return best


def automl_predict(c: dict, name: str, targets) -> np.ndarray:
    """Refit the selected candidate on all dev trials and predict `targets`."""
    if not name:
        raise ValueError(f"no AutoML candidate was scored for case {c['id']}")
    _, bin_size, make = next(x for x in automl_candidates() if x[0] == name)
    x = binned_counts(c["dev"], c["n_units"], bin_size)
    return make().fit(x, c["y"]).predict(binned_counts(targets, c["n_units"], bin_size))


# What the released POYO+ checkpoint's pickle refers to (checked before first use); anything else is refused.
_CKPT_GLOBALS = {("collections", "OrderedDict"), ("collections", "defaultdict"), ("typing", "Any"), ("typing", "Dict"),
                 ("torch._utils", "_rebuild_tensor_v2"), ("torch", "FloatStorage"), ("numpy", "dtype"),
                 ("numpy._core.multiarray", "scalar"), ("_codecs", "encode"), ("omegaconf.dictconfig", "DictConfig"),
                 ("omegaconf.listconfig", "ListConfig"), ("omegaconf.base", "ContainerMetadata"),
                 ("omegaconf.base", "Metadata"), ("omegaconf.nodes", "AnyNode"),
                 *((m, n) for m in ("builtins", "__builtin__") for n in ("dict", "list", "int", "str", "long", "unicode"))}


def load_checkpoint(path: Path) -> dict:
    """torch.load of a checkpoint that carries its run config (so weights_only cannot read it),
    resolving only the globals in _CKPT_GLOBALS."""
    import pickle
    import types

    import torch

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in _CKPT_GLOBALS:
                raise pickle.UnpicklingError(f"checkpoint refers to {module}.{name}")
            return super().find_class(module, name)

    return torch.load(path, map_location="cpu", weights_only=False,
                      pickle_module=types.SimpleNamespace(Unpickler=Unpickler, load=pickle.load, __name__="allowlist"))


def per_sample_sdpa(*, query, key, value, q_pos_emb, kv_pos_emb, q_seqlen, kv_seqlen, num_heads, dropout_p,
                    rotate_value):
    """ibl-bwb's core/nn/varlen_attention.py:_nested_sdpa with its one jagged SDPA call, which has
    kernels on CUDA only, replaced by a dense SDPA per sample: the same block-diagonal attention."""
    import torch
    import torch.nn.functional as F
    from torch_brain.nn import RotaryTimeEmbedding

    query, key, value = (x.unflatten(-1, (num_heads, -1)) for x in (query, key, value))  # (N, H, D)
    query = RotaryTimeEmbedding.rotate(x=query, rotary_emb=q_pos_emb, unsqueeze_dim=1)
    key = RotaryTimeEmbedding.rotate(x=key, rotary_emb=kv_pos_emb, unsqueeze_dim=1)
    if rotate_value:
        value = RotaryTimeEmbedding.rotate(x=value, rotary_emb=kv_pos_emb, unsqueeze_dim=1)
    qs, ks = q_seqlen.tolist(), kv_seqlen.tolist()
    if len(set(qs)) == 1 and len(set(ks)) == 1:  # equal lengths (latents, outputs): one batched call
        q, k, v = (x.unflatten(0, (len(qs), -1)).transpose(1, 2) for x in (query, key, value))  # (B, H, n, D)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p).transpose(1, 2).flatten(0, 1)
    else:
        out = torch.cat([F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                                                        dropout_p=dropout_p).transpose(0, 1)
                         for q, k, v in zip(query.split(qs), key.split(ks), value.split(ks))])
    if rotate_value:
        out = RotaryTimeEmbedding.rotate(x=out, rotary_emb=RotaryTimeEmbedding.invert(q_pos_emb), unsqueeze_dim=1)
    return out.flatten(-2)


def _ibl_bwb(src: Path):
    """POYOPlus and the helpers its finetuning uses, from the ibl-bwb source, skipping the package
    __init__s that import every model and the training stack. Off CUDA, attention runs through
    per_sample_sdpa."""
    import sys
    import types

    import torch

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    for name in ("pretrain", "pretrain.models", "pretrain.models.poyo_plus"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            sys.modules[name].__path__ = [str(Path(src, *name.split(".")))]
    from core.nn import varlen_attention
    from core.optim import param_groups
    from ibl_bwb_eval.tasks import get_ts1_readout_spec
    from pretrain.models.poyo_plus.poyo_plus import POYOPlus
    if not torch.cuda.is_available():
        varlen_attention._nested_sdpa = per_sample_sdpa
    return POYOPlus, get_ts1_readout_spec("wheel_speed"), param_groups


def poyo_finetune(ckpt: dict, src: Path, train, y_train, val, y_val, targets, n_units: int, *, base_lr: float,
                  seed: int, device, frame_times=None, name: str = "case", num_epochs: int = 300, batch_size: int = 32,
                  micro_batch: int = 32, epoch_log=None):
    """POYO+ from the pretrained checkpoint, finetuned on one session and used to predict `targets`,
    as `trainer=poyo_plus_gradual_unfreezing task=wheel_speed` does it (POYOPlusEvalTrainer on
    TS1EvalTrainer; ts1/configs/train.yaml): the session's units and the session get new
    embeddings, and only the unit and session embeddings train until epoch 40, then everything;
    UnitDropout (300 to 1800 units, mode 800) on train samples; MSE; AdamW (weight decay 1e-4
    except biases, norms and embeddings) under OneCycleLR over 300 epochs; gradient norm clipped
    at 1; batches of 32, the last partial one dropped; val R^2 after every epoch, early stopping
    with patience 20 from epoch 50; the best-val weights predict. Differences: fp32 (the benchmark
    autocasts to bf16 on CUDA), the 'nested' attention kernel (same weights and values as
    xformers), and the wheel frames are queried at `frame_times` (seconds since onset, one row of
    50 per trial for train, val and targets), by default the frame-interval midpoints because the
    staged data do not carry each trial's frame phase. `micro_batch` < 32 accumulates gradients
    over equal parts of each batch (the same mean loss) to fit in memory.
    Returns (best val R^2, predictions for `targets`, (n, 50))."""
    import copy
    import random
    import types

    import torch
    import torch.nn.functional as F
    from torch_brain.batching import collate
    from torch_brain.data import ArrayDict, Data, Interval, IrregularTimeSeries
    from torch_brain.transforms import UnitDropout

    POYOPlus, spec, param_groups = _ibl_bwb(src)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    unit_ids, session = np.array([f"nabench/{name}/{i}" for i in range(n_units)]), f"nabench/{name}"
    mid = (np.arange(N_FRAMES) + 0.5) / FS
    ft = frame_times or tuple(np.tile(mid, (len(s), 1)) for s in (train, val, targets))
    model_cfg = {k: v for k, v in ckpt["cfg"]["model"].items() if k != "_target_"} | {"attn_impl": "nested"}
    model = POYOPlus(**model_cfg)
    ds = types.SimpleNamespace(context_window=1.0, get_unit_ids=lambda: list(unit_ids), get_session_ids=lambda: [session])
    model.link_datasets(ds, ds, ds, finetune_enable=True)
    model.configure_readout(spec)
    model.load_ckpt(ckpt)
    model.to(device)
    dropout = UnitDropout(max_units=1800, min_units=300, mode_units=800, peak=4)

    def batch(trials, ys, fts, rows, augment):
        xs, ts = [], []
        for r in rows:
            (t, u), y, f = trials[r], ys[r], fts[r]
            d = Data(spikes=IrregularTimeSeries(timestamps=t.astype(np.float64), unit_index=u.astype(np.int64),
                                                domain=Interval(0.0, 1.0)),
                     units=ArrayDict(id=unit_ids), session=Data(id=session),
                     wheel=IrregularTimeSeries(timestamps=f.astype(np.float64), wheel_speed=y.astype(np.float32),
                                               domain=Interval(0.0, 1.0)), domain=Interval(0.0, 1.0))
            xs.append(model.input_fn(dropout(d) if augment else d))
            ts.append({"values": y.astype(np.float32), "timestamps": f.astype(np.float32)})
        X, Y = collate(xs), collate(ts)
        to = lambda x: x.to(device, torch.float32) if x.is_floating_point() else x.to(device)  # noqa: E731 (no float64 on MPS)
        inputs = {k: to(v) for k, v in X["model_inputs"].items() if k not in ("output_timestamps", "output_session_index")}
        pred = model(**inputs, output_timestamps=to(Y["timestamps"]), output_session_index=to(X["session_index"]),
                     return_dict=False)
        return pred.reshape(len(rows), -1), to(Y["values"]).reshape(len(rows), -1)

    def predict(trials, fts):
        model.eval()
        zeros = np.zeros((len(trials), N_FRAMES), np.float32)
        with torch.no_grad():
            out = [batch(trials, zeros, fts, range(i, min(i + micro_batch, len(trials))), False)[0].cpu().numpy()
                   for i in range(0, len(trials), micro_batch)]
        return np.concatenate(out) if out else zeros

    groups = param_groups(model, weight_decay=1e-4, no_weight_decay=("bias", "norm", "emb"))
    opt = torch.optim.AdamW(groups, lr=base_lr)
    steps = len(train) // batch_size
    if steps == 0:
        raise ValueError(f"batch size {batch_size} > {len(train)} train trials")
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=base_lr, epochs=num_epochs, steps_per_epoch=steps,
                                                pct_start=0.5, anneal_strategy="cos", div_factor=1.0, final_div_factor=1e4)
    frozen = [p for n, p in model.named_parameters() if not n.startswith(("unit_emb.", "session_emb."))]
    for p in frozen:
        p.requires_grad = False
    best, best_state, patience = -np.inf, None, 20
    for epoch in range(num_epochs):
        if epoch == 40:
            for p in frozen:
                p.requires_grad = True
        model.train()
        perm = torch.randperm(len(train), generator=gen).tolist()
        for s in range(steps):
            rows = perm[s * batch_size:(s + 1) * batch_size]
            opt.zero_grad()
            for i in range(0, batch_size, micro_batch):
                pred, y = batch(train, y_train, ft[0], rows[i:i + micro_batch], True)
                (F.mse_loss(pred, y) * (len(pred) / batch_size)).backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
            opt.step()
            sched.step()
        score = pooled_r2(predict(val, ft[1]), y_val)
        if np.isfinite(score) and score > best:
            best, patience = score, 20
            best_state = {k: v.detach().to("cpu", copy=True) if torch.is_tensor(v) else copy.deepcopy(v)
                          for k, v in model.state_dict().items()}
        elif epoch >= 50:
            patience -= 1
        if epoch_log:
            epoch_log(epoch, score)
        if patience <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, predict(targets, ft[2])


def _published(root: Path, name: str, cid: str) -> dict:
    import json
    return json.loads((root / "reference" / name).read_text())["cases"][cid]["params"]


def _simple(cases, workspace, budget, root):
    """Linear: per session the selected configuration, trained at the 5 eval seeds; the mean of their predictions."""
    out = {}
    for c in cases:
        tr, va = ~c["is_val"], c["is_val"]
        args = (_pick(c["dev"], tr), c["y"][tr], _pick(c["dev"], va), c["y"][va])
        params = (_published(root, "linear_configs.json", c["id"]) if SELECTION == "published"
                  else linear_sweep(*args, c["n_units"]))
        out[c["id"]] = np.mean([linear_fit(*args, c["test"], c["n_units"], **params, seed=s)[1] for s in EVAL_SEEDS], 0)
    return out


def _automl(cases, workspace, budget, root):
    """automl_select, then the refits; what was searched and chosen goes to workspace/automl_search.json."""
    import json
    import time

    t0, searched = time.time(), []
    best = automl_select(cases, AUTOML_SELECTION_SHARE * float(budget["wall_time_s"]), searched)
    t1 = time.time()
    out = {c["id"]: automl_predict(c, best[c["id"]][0], c["test"]) for c in cases}
    (Path(workspace) / "automl_search.json").write_text(json.dumps(
        {"selection_s": t1 - t0, "refit_s": time.time() - t1, "searched": searched, "chosen": best}, indent=1))
    return out


def _sota(cases, workspace, budget, root, cache: Path | None = None):
    """POYO+ finetuned per session at the selected learning rate, at the first SOTA_SEEDS eval seeds; writes each
    seed's predictions to workspace/seeds/seed<k>.csv and returns their mean. Run by the GPU job
    (scripts/sota_gpu_job.py); each finished fit is kept in `cache`, so a rerun resumes. The fixed protocol, not the
    wall-time budget, sets its length."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # MPS cannot run it (see the report)
    ckpt = load_checkpoint(root / "tools" / "poyo_plus" / "checkpoint.pt")
    src = Path(root).parents[1] / "src" / "ibl-bwb" / "src"
    per_seed = {k: {} for k in range(1, SOTA_SEEDS + 1)}
    for c in cases:
        tr, va = ~c["is_val"], c["is_val"]
        args = (_pick(c["dev"], tr), c["y"][tr], _pick(c["dev"], va), c["y"][va])
        fit = lambda targets, **kw: poyo_finetune(ckpt, src, *args, targets, c["n_units"], device=device, name=c["id"], **kw)  # noqa: E731
        lr = (_published(root, "poyo_plus_lrs.json", c["id"])["base_lr"] if SELECTION == "published"
              else max(POYO_LRS, key=lambda x: fit([], base_lr=x, seed=POYO_SWEEP_SEED)[0]))
        for k in per_seed:
            part = Path(cache) / f"{c['id']}_seed{k}.npy" if cache else None
            if part and part.exists():  # every fit seeds itself, so a resumed run is the same as an unbroken one
                per_seed[k][c["id"]] = np.load(part)
                continue
            per_seed[k][c["id"]] = fit(c["test"], base_lr=lr, seed=EVAL_SEEDS[k - 1])[1]
            if part:
                np.save(part, per_seed[k][c["id"]])
    (Path(workspace) / "seeds").mkdir(exist_ok=True)
    for k, preds in per_seed.items():
        write_cases(Path(workspace) / "seeds" / f"seed{k}.csv", {cid: _fmt(p) for cid, p in preds.items()})
    return {c["id"]: np.mean([per_seed[k][c["id"]] for k in per_seed], 0) for c in cases}


def _fmt(pred: np.ndarray) -> str:
    return " ".join(f"{v:.7g}" for v in np.asarray(pred, np.float32).ravel())


def _task_root() -> Path:
    from ..config import load_config

    # shortcut: the default config's data_root; a batch run with another --config must keep the same data_root
    return Path(load_config()["paths"]["data_root"]) / "tasks" / "bwb_neuropixels_wheel"


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _manifest(path: Path) -> dict[str, str]:
    """sha256sum's format: '<hash>  <relative path>' per line."""
    return {rel: h for h, rel in (line.split("  ", 1) for line in Path(path).read_text().splitlines() if line)}


def write_manifest(root: Path, files) -> None:
    Path(root, "MANIFEST.sha256").write_text("".join(f"{_sha256(Path(root, f))}  {f}\n" for f in sorted(files)))


SOTA_RESULT = ("predictions.csv", "provenance.json", *(f"seeds/seed{k}.csv" for k in range(1, SOTA_SEEDS + 1)))


def check_sota_result(result: Path, data: Path) -> list[str]:
    """What is wrong with a GPU SOTA job's output (scripts/sota_gpu_job.py) for the inputs in `data` (a
    workspace's data/ or the task's data_root folder): files missing or not matching its manifest, inputs other
    than these, or predictions without every case and 50 finite frames per test trial, or an ensemble that is
    not the mean of the seeds. Reads no labels."""
    import json

    from ..metrics import read_cases

    result, data = Path(result), Path(data)
    try:
        listed = _manifest(result / "MANIFEST.sha256")
    except (OSError, ValueError) as e:
        return [f"no readable MANIFEST.sha256: {e}"]
    errors = [f"{f}: not in the manifest" for f in SOTA_RESULT if f not in listed]
    errors += [f"{f}: missing or checksum differs" for f in SOTA_RESULT
               if f in listed and not ((result / f).is_file() and _sha256(result / f) == listed[f])]
    if errors:
        return errors
    seen = json.loads((result / "provenance.json").read_text()).get("inputs_sha256", {})
    errors += [f"{rel}: the job read a different file" for rel in SOTA_INPUTS if seen.get(rel) != _sha256(data / rel)]
    trials = np.loadtxt(data / "test" / "trial_index.csv", delimiter=",", skiprows=1, dtype=str, ndmin=2)[:, 0]
    want = {c: N_FRAMES * int((trials == c).sum()) for c in read_cases(data / "test" / "case_ids.csv", "case_id")}
    preds = {}
    for f in (f for f in SOTA_RESULT if f.endswith(".csv")):
        rows = read_cases(result / f, "prediction")
        if set(rows) != set(want):
            errors.append(f"{f}: its cases are not those in test/case_ids.csv")
            continue
        try:
            preds[f] = {c: _trace(v) for c, v in rows.items()}
        except ValueError:
            errors.append(f"{f}: values that are not numbers")
            continue
        bad = sum(len(preds[f][c]) != n or not np.isfinite(preds[f][c]).all() for c, n in want.items())
        if bad:
            errors.append(f"{f}: {bad} case(s) without {N_FRAMES} finite values per test trial")
    if not errors:
        mean = {c: np.mean([preds[f][c] for f in preds if f.startswith("seeds/")], 0) for c in want}
        off = sum(not np.allclose(preds["predictions.csv"][c], mean[c], rtol=1e-5, atol=1e-6) for c in want)
        if off:
            errors.append(f"predictions.csv: not the mean of the seeds for {off} case(s)")
    return errors


def _child(kind: str, workspace: Path, budget: dict) -> dict[str, str]:
    """Run one baseline in the task's environment and read back its predictions."""
    import json
    import subprocess
    import tempfile

    from ..metrics import read_cases

    python = _task_root().parents[1] / "envs" / "bwb" / "bin" / "python"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "predictions.csv"
        subprocess.run([str(python), "-I", "-m", __name__, kind, str(workspace), json.dumps(budget), str(out)], check=True)
        return read_cases(out, "prediction")


def simple_baseline(workspace: Path, budget: dict) -> dict[str, str]:
    """The benchmark's Linear baseline: see linear_fit and _simple."""
    return _child("simple", workspace, budget)


def sota_baseline(workspace: Path, budget: dict) -> dict[str, str]:
    """POYO+ pretrained on the 423 pretraining sessions and finetuned per session (poyo_finetune, _sota), run on a
    GPU machine and imported to data_root (D-010, scripts/sota_gpu.py): the 5-seed ensemble, checked against the
    staged inputs. Each seed's predictions go to workspace/seeds/ for the reference rows."""
    import shutil

    from ..metrics import read_cases
    from ..runner import NotReady

    src = _task_root() / "sota_gpu"
    if not src.exists():
        raise NotReady(f"no GPU SOTA predictions imported to {src} yet (scripts/sota_gpu.py)")
    errors = check_sota_result(src, Path(workspace) / "data")
    if errors:
        raise ValueError(f"imported SOTA predictions in {src} unusable: {'; '.join(errors)}")
    shutil.copytree(src / "seeds", Path(workspace) / "seeds")
    return read_cases(src / "predictions.csv", "prediction")


def automl_baseline(workspace: Path, budget: dict) -> dict[str, str]:
    """Model and hyperparameter search per session within the wall-time budget, selected on the is_val trials
    and refitted on all dev trials; candidates in automl_candidates. An AutoML anchor, not a tuned SOTA."""
    return _child("automl", workspace, budget)


if __name__ == "__main__":
    import json
    import sys

    kind, ws, budget, out = sys.argv[1:5]
    run = {"simple": _simple, "automl": _automl}[kind]
    write_cases(out, {cid: _fmt(p) for cid, p in run(load_cases(Path(ws)), Path(ws), json.loads(budget), _task_root()).items()})
