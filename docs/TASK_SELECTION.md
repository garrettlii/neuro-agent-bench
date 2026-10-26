# Task selection

## Rule

A task qualifies for the main suite only if all of the following hold:

a. **Hidden test labels.** Its test labels are hidden, or were released after every evaluated model's knowledge cutoff, or a private re-split from never-released subjects or sessions is possible.
b. **Runnable SOTA.** A published SOTA with runnable code exists.
c. **Compute.** One run needs at most about one GPU-day.
d. **Coverage.** It adds modality coverage (fMRI, EEG/MEG, spikes, calcium imaging, behaviour, BCI).

Every candidate considered goes in the registry below with a decision and a reason, including rejections. Decisions are made before any agent runs on the task.

## Pilot tasks

The five pilot tasks were chosen before this rule existed. They are listed as "pilot, chosen before the rule", carry `selection: pilot_pre_rule` in their specs, and are reported separately from rule-selected tasks.

## Registry

| Task | Modality | (a) labels | (b) SOTA code | (c) ≤ 1 GPU-day | (d) coverage | Decision | Reason | Date |
|---|---|---|---|---|---|---|---|---|
| niclip_map_to_paper | fMRI | FILL | FILL | FILL | FILL | pilot, chosen before the rule | pilot; audit in `audits/niclip_map_to_paper.md` | 2026-10-26 |
| nsd_cross_subject_id | fMRI | FILL | FILL | FILL | FILL | pilot, chosen before the rule | pilot; audit in `audits/nsd_cross_subject_id.md` | 2026-10-26 |
| stop_signal_behaviour | behaviour | FILL | FILL | FILL | FILL | pilot, chosen before the rule | pilot; audit in `audits/stop_signal_behaviour.md` | 2026-10-26 |
| bwb_neuropixels_wheel | spikes | FILL | FILL | FILL | FILL | pilot, chosen before the rule | pilot; audit in `audits/bwb_neuropixels_wheel.md` | 2026-10-26 |
| t15_speech | BCI | FILL | FILL | FILL | FILL | pilot, chosen before the rule | pilot; audit in `audits/t15_speech.md` | 2026-10-26 |

`dummy_spikes` is a synthetic harness test, not a candidate.
