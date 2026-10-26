# Data

Everything lives outside the repository under `data_root` (inputs, tool weights) and `truth_root` (test labels). SHA-256 of every staged file is recorded in each batch manifest, and of each truth file in the truth-side manifest.

| Task | Source | URL | Licence | Data released | Test labels released | Local path | SHA-256 |
|---|---|---|---|---|---|---|---|
| dummy_spikes | simulated (`scripts/build_dummy.py`, fixed seed) | n/a | this repository | 2026-10-26 | never | `data_root/tasks/dummy_spikes/` | per batch manifest |
| niclip_map_to_paper | FILL | FILL | FILL | FILL | FILL | `data_root/tasks/niclip_map_to_paper/` | FILL |
| nsd_cross_subject_id | FILL | FILL | FILL | FILL | FILL | `data_root/tasks/nsd_cross_subject_id/` | FILL |
| stop_signal_behaviour | FILL | FILL | FILL | FILL | FILL | `data_root/tasks/stop_signal_behaviour/` | FILL |
| bwb_neuropixels_wheel | FILL | FILL | FILL | FILL | FILL | `data_root/tasks/bwb_neuropixels_wheel/` | FILL |
| t15_speech | FILL | FILL | FILL | FILL | FILL | `data_root/tasks/t15_speech/` | FILL |

For each real dataset, record the exact download (version, files) and the SHA-256 of each raw file here when it is fetched.
