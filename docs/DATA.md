# Data

Everything lives outside the repository under `data_root` (inputs, tool weights) and `truth_root` (test labels). SHA-256 of every staged file is recorded in each batch manifest, and of each truth file in the truth-side manifest.

| Task | Source | URL | Licence | Data released | Test labels released | Local path | SHA-256 |
|---|---|---|---|---|---|---|---|
| dummy_spikes | simulated (`scripts/build_dummy.py`, fixed seed) | n/a | this repository | 2026-10-26 | never | `data_root/tasks/dummy_spikes/` | per batch manifest |
| niclip_map_to_paper | FILL | FILL | FILL | FILL | FILL | `data_root/tasks/niclip_map_to_paper/` | FILL |
| nsd_cross_subject_id | Natural Scenes Dataset, NSD-core (Allen et al. 2022, Nature Neuroscience) | https://naturalscenesdataset.org (`s3://natural-scenes-dataset`) | NSD Terms and Conditions: research and education only, no redistribution of the data or its components | 2021-09-03; last three core sessions per subject 2023-08-20 | FILL | `data_root/tasks/nsd_cross_subject_id/` | FILL |
| stop_signal_behaviour | FILL | FILL | FILL | FILL | FILL | `data_root/tasks/stop_signal_behaviour/` | FILL |
| bwb_neuropixels_wheel | IBL BrainWideBench TS1 (arXiv 2609.22064v1), built on the IBL Brain Wide Map (DANDI 000409) | https://github.com/brainbench-org/ibl-bwb (`s3://brain-wide-bench`) | raw data CC-BY-4.0 (DANDI 000409); processed build: none stated; code MIT | FILL | FILL | `data_root/tasks/bwb_neuropixels_wheel/` | FILL |
| t15_speech | Brain-to-Text '25, participant T15 (Dryad doi:10.5061/dryad.dncjsxm85) | https://www.kaggle.com/competitions/brain-to-text-25 | CC0 1.0 | FILL | FILL | `data_root/tasks/t15_speech/` | FILL |

For each real dataset, record the exact download (version, files) and the SHA-256 of each raw file here when it is fetched.

## bwb_neuropixels_wheel

**Source.** The benchmark's own processed evaluation sessions: `s3://brain-wide-bench/brainsets/all_units/ibl_brain_wide_bench_2026/eval/`, 29 `.h5` files (49.1 GB), brainset `derived_version` 0.1.0, `unit_filtering` all_units, built and uploaded to S3 on 2026-09-21, read anonymously over HTTPS on 2026-10-27. These are the files the TS1 loader at ibl-bwb @ 6fbd666 reads; rebuilding from DANDI 000409 would not reproduce them (the pipeline reads ONE/openalyx with its own resampling). The build states no licence, so the raw files stay under `data_root/raw/bwb/`, the arrays derived from them are staged only into local workspaces, and neither is redistributed. The per-file SHA-256 and S3 ETags are in `data_root/raw/bwb/MANIFEST.txt`, outside the repo because the file names are evaluation session ids; SHA-256 of that manifest: `6b9dc5b12dae0e0784bbf301e861da6e2ad47000a7670be8152155c0b81e1802` (fetched 2026-10-27). The 29 files' SHA-256, sorted by value and without file names, are listed after the loading example below. All 29 files carry the same provenance block: pipeline commit 2bec8d49 (pre-publication, not in the public history), clean tree, `ONE_REVISION_LAST_BEFORE` 2026-09-03, built 2026-09-21. The build matches arXiv 2609.22064v1 Table A6 exactly: 43,187 units and 7,795 / 3,471 / 5,894 train / val / test windows (294,700 test frames).

**Build.** In an environment with h5py (for example a venv under `data_root/envs/`, with this package installed): `python -I scripts/build_bwb_neuropixels_wheel.py`. It writes the files below and `truth_root/tasks/bwb_neuropixels_wheel/case_map.csv` (case id to session id, truth side only). Case ids `s01`..`s29` are assigned in a seeded random order: `numpy.random.default_rng(20261027).permutation` of the files sorted by name. Trials are renumbered 0, 1, 2, … per session in time order, dev and test as one sequence (dev 0 to n_dev − 1, test from n_dev), so the numbers do not reveal excluded or gap trials. The build also writes, under `data_root/tasks/bwb_neuropixels_wheel/`, files that are never staged: `scramble/<case>.npz`, the inputs of the C3 scramble (the session's full spike train, its time span, every dev and test window in staged order and the benchmark's trial numbers), 17 GB.

**Samples.** One per trial, as the benchmark's TS1 loader serves `wheel_speed` at its default context (`context_length` 1.0 s): the window [movement onset, movement onset + 1 s), holding every spike of every unit of the session in that window and the 50 frames of the 50 Hz wheel-speed series (rad/s, raw, not normalized) whose grid points fall in it. Dev trials are the benchmark's train and validation trials, test trials its test trials; trials in the 20 s gaps between splits are in neither. All trials are 1.0 s; the build refuses a window that a split or signal edge would clip. Each sample was checked against the benchmark's own `IBLBrainWideBenchTS1` with `TrialSampler` for all 29 sessions: same trials in the same order, same spikes and units, bit-identical float32 frames. Spike times are float32; the few that float32 would round onto the window end (607 of 187 M) are kept just below it, under 1e-7 s from their float64 value.

| File | Key or column | dtype, shape | Content |
|---|---|---|---|
| `dev/sessions.npz`, `test/sessions.npz` | `<case>_n_units` | int64, () | units in the session; unit indices are 0 .. n_units−1 |
| | `<case>_trial_index` | int64, (n_trials,) | trial number in the session, 0, 1, … in time order, test continuing after dev; not the benchmark's trial numbers |
| | `<case>_spike_ptr` | int64, (n_trials + 1,) | spikes of trial i are entries `ptr[i]:ptr[i+1]` of the next two arrays |
| | `<case>_spike_times` | float32, (n_spikes,) | seconds since the trial's movement onset, in [0, 1) |
| | `<case>_spike_units` | int32, (n_spikes,) | unit index of each spike |
| `dev/sessions.npz` only | `<case>_is_val` | bool, (n_trials,) | True for the benchmark's validation trials (after its train trials and a 20 s gap) |
| `dev/wheel_speed.npz` | `<case>` | float32, (n_trials, 50) | wheel speed at the 50 frames of each dev trial, rows in `<case>_trial_index` order |
| `test/trial_index.csv` | `case_id,trial_index` | | one row per test trial, sessions in `case_ids.csv` order, trials in time order |
| `test/case_ids.csv` | `case_id` | | one row per session |
| `truth_root/.../test_labels.csv` | `case_id,label` | | 50 values per test trial in `trial_index.csv` order, space-separated (`%.9g` of the float32 values) |
| `scramble/<case>.npz` (never staged) | `times`, `units`, `span`, `dev_windows`, `test_windows`, `source_trial_index` | float64, int32, (2,), (n, 2), (n, 2), int64 | full spike train, spike-domain start and end, each staged trial's [onset, onset + 1 s), benchmark trial numbers (dev then test) |

```python
z, y = np.load("data/dev/sessions.npz"), np.load("data/dev/wheel_speed.npz")
ptr = z["s01_spike_ptr"]
t, u = z["s01_spike_times"][ptr[i]:ptr[i + 1]], z["s01_spike_units"][ptr[i]:ptr[i + 1]]   # trial i
speed = y["s01"][i]   # its 50 frames
```

**Baseline inputs (harness only, never staged).** `sota_gpu/` holds the imported output of the GPU SOTA job (D-010, `scripts/sota_gpu.py import`): the 5-seed ensemble, each seed's predictions, the job's provenance (the SHA-256 of the inputs it read, device, versions, wall time) and its manifest. `tools/poyo_plus/checkpoint.pt` is the released POYO+ checkpoint, `nerdslab/ibl-bwb-poyo_plus` on the Hugging Face Hub at revision `6e717aad42a6094bb2691721f31ffcdfa8d5cd97` (MIT; SHA-256 `2eca8d45b6783e62a869ed586d02e64c6070d13e482083070e3a240ce10e4c92`, equal to the Hub's LFS id; downloaded 2026-10-27). Its stored run config and vocabularies show what it was trained on: W&B run `iy81kqrk`, pretraining code commit 17a5d13b, processed data v0.0.9, best checkpoint at epoch 68; its session vocabulary is exactly the 423 ids of `src/core/data/pretrain_recording_ids.txt` and contains none of the 29 evaluation ids, and none of the 43,187 evaluation unit ids is in its unit vocabulary. `reference/linear_configs.json` and `reference/poyo_plus_lrs.json` hold, per case, the hyperparameters the benchmark selected on that session's validation trials and its per-seed validation R², taken on 2026-10-27 from the public W&B projects `ibl-bwb/ts1-linear-ss` (145 wheel_speed runs: 29 sessions × eval seeds 43-47) and `ibl-bwb/ts1-poyo_plus-ft-seeds` (the same; learning rate chosen in `ibl-bwb/ts1-poyo_plus-ft`, 5 rates at seed 42); only configs and validation metrics were kept. Those runs used processed data v0.0.11 (Linear) and v0.0.9 (POYO+), not the public v0.1.0 build.

SHA-256 of the 29 evaluation `.h5` files, sorted by value:

```
122ac8411c507e99c7987d68a2b6e26bdf255076c7a60ab3bbbc60a6686ab659
1294543e745a585b231c6a72c369376c5120b015a98fca12a37ea53f6e1b846d
16900301b7c3e02a9cb0c5b6d632f938adc63952d06f13f566f29eb4878059d4
172147b5cdd409c85b108e5bfff44c357ca9b78ba1b6c1ea706915242a12ea58
228876c9da79ed9f7a712aebc19ad36dca9adf164146e2dd4a35e358d6d48ce3
265f836013c321fc26b80ab57514447d28a1ce9c1ec8c62c99492abd97c47056
28c07feac2464886d84c0cb1376de7347f04d4134d0d55fc6306c7c861ac40c4
341405ab1f6695d452409cf13b17de67f96b7441f073a8bdfd95d23a9debd218
3b45d8e7a7c1f501ff4f4e254730f5e642842e39504316f484aebdcf3ce7a142
4c8c4fdd84548e520398b6ace37dcccf5f13978e9db43209f4fe19233906d5e8
53bdaa249c83444e3b3daa1a5dac3e8c88610bff2ef7386843b472c039b21fd2
650be91c26eefe5e20fc8edcdb703f993ec32b963b012924a86a76d13e3ea64e
65fbf78f687f6c8b00931c465c76689066fb6d596cd727166fad5918b64c4f09
6794766fbd88008c125a0028d2a8ab8c23778c2c1b407b23f4ee47570bebce84
67ba228c0f488215a5d7c8bc386ecce0df39011ecb0a3ffe98915c664605d92d
79802d5b9ec7c6ad65c39527b01ab40a5e02489064efe7f326c088c4f84fcd48
81ebee1c9c4773c260af47229297a81f78c0d3f224143a5a37b449e27d98b188
8a0ff01357c01713d7d88b297a4222847376e867be836ea9b0efe5e6323cea8e
8d06ebb47984af8478bdd37c978640f01d571a9ec53824fa99f223b41f3366ed
95c9e8e72624e85981f8bd68e9d98629b5a6e80d582abb6d61dc2f401a6b6a3e
98a0adc077be4546147f639576d088ae428239f3fb8be0f89cf02b4357017e28
9be47f27f460c3cbd51a5cc5304f7e391995757b5bcdc8805d57f18195954fea
a9cdc3886d013979c7b4ae548aa683e8db0b38b9501c498f51013c7215f7511b
acbb277b5cf01706558b3a4e18fcf2221b6245ae67676b29b7b5bb4a45f94299
b5e003d7e46a9b931b10f5a76b454dc868afdc72d3c0d99164595b3efcaedd40
eadcc0efff65452b96ffc0edcd03382f1086f8ab617fcb85e4c75d06a0c8aa45
efeefbf0801b12d49ed6a90daa051d440b17da2e406176e4bd2541045c8438bb
f1cc0204c5e2158c49d78f41a4bfb54d9f54dc2a84e834d60c0cbef39b312e56
f55d2c1cfda2e2b9642ad824160d504526e27c64aad981ac1d358be259091ff4
```
