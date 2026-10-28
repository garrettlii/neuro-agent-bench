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
| niclip_map_to_paper | fMRI | FILL: source corpus not yet identified | yes: NiCLIP code is public (NBCLab/brain-decoder, Apache-2.0); not yet run here | FILL | partly: fMRI, also covered by nsd_cross_subject_id | pilot, chosen before the rule | pilot; audit in `audits/niclip_map_to_paper.md` | 2026-10-26 |
| nsd_cross_subject_id | fMRI | no: all NSD-core trials are public (last withheld sessions released 2023-08-20), so no never-released sessions remain | FILL: SOTA decoder not yet identified | FILL | partly: fMRI, also covered by niclip_map_to_paper | pilot, chosen before the rule | pilot; audit in `audits/nsd_cross_subject_id.md` | 2026-10-26 |
| stop_signal_behaviour | behaviour | FILL: dataset not yet named | FILL: dataset not yet named | FILL | yes: only behaviour task | pilot, chosen before the rule | pilot; audit in `audits/stop_signal_behaviour.md` | 2026-10-26 |
| bwb_neuropixels_wheel | spikes | no: all sessions and wheel traces are public (arXiv 2609.22064v1 p. 27; DANDI 000409), so no private re-split | yes: POYO+ code (brainbench-org/ibl-bwb, MIT) and checkpoints are published; not yet run here | FILL | yes: only spikes task | pilot, chosen before the rule | pilot; audit in `audits/bwb_neuropixels_wheel.md` | 2026-10-26 |
| t15_speech | BCI | yes: competition test transcripts were not released with the data (Dryad record, organizers' README); a later release on Kaggle is unverified | FILL: top entry with released code not yet identified | FILL | yes: only BCI task | pilot, chosen before the rule | pilot; audit in `audits/t15_speech.md` | 2026-10-26 |
| niclip_post_cutoff | fMRI | FILL: depends on every source paper's first publication date against every evaluated model's cutoff | FILL: same SOTA as niclip_map_to_paper; not yet run here | FILL | partly: fMRI, also covered by nsd_cross_subject_id and niclip_map_to_paper | candidate | map-to-paper identification built only from papers first published after 2026-07-01, maps built with the same MKDA pipeline as niclip_map_to_paper | 2026-10-27 |

`dummy_spikes` is a synthetic harness test, not a candidate.
