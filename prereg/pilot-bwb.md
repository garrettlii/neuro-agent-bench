# Pre-registration: pilot-bwb-1

- Authors: Garrett Li
- Date: 2026-10-27
- Tasks and their frozen spec commits: bwb_neuropixels_wheel, frozen in the commit that adds this file
- Models, CLI versions and knowledge cutoffs: opus-5.5-cli (Claude Code 2.1.312, cutoff 2026-07-01); sol-6.1-cli (Codex CLI 0.163.0, cutoff 2026-05-01)
- This pilot runs two model families and no open-weights model, so it does not meet the design's open-weights requirement (D5); the report states this.
- C1 is deferred: on this machine the POYO+ tool runs on CPU only. SOTA ran on a GPU and agents on CPU, which disadvantages agents.

## Hypotheses

Scores below are normalized against the SOTA rerun: (agent − simple) / (SOTA − simple), where simple is the single-session linear decoder (published R² 0.219) and SOTA is POYO+ pretrained on 423 sessions and finetuned per evaluation session (published R² 0.362). On published numbers the single-session CNN and MLP sit at about 0.78, and the earlier, unaudited pilot result (R² 0.305) at about 0.60.

**H0, SOTA reproduces.** The POYO+ rerun on a separate GPU machine, with 5 finetuning seeds, lands close to the published 0.362. If it does not, every normalized score shifts, so this is checked first.

**H2, without trained models, agents reach strong single-session baselines but not SOTA (C2).** POYO+'s advantage comes from pretraining on 423 sessions that the agent never sees. An agent can pool the 29 evaluation sessions' dev data, which may lift it above single-session baselines, but not to the pretrained model within the budget.

**H4, no leakage (C3).** With spikes circularly shifted against behaviour, the only learnable structure left is the stereotyped time course of wheel speed after movement onset. C3 should score no better than a predictor that uses that time course alone.

**H5, programs written by agents do about as well as AutoML at the same budget (C2 vs AutoML).**

**H6, the two model families are not distinguishable on this task at this sample size.** With 29 sessions as cases, differences smaller than about 0.15 normalized are within noise.

## Analysis

- Primary outcome: normalized score vs SOTA per task and condition, paired bootstrap over test cases (`docs/DESIGN.md`). Cases are the 29 evaluation sessions; reps are averaged per case before bootstrapping.
- Secondary: normalized score vs noise ceiling (none for this task); costs.
- Signal-free reference: the dev-set mean wheel-speed time course after movement onset, applied to every test trial, scored with the task metric by the scorer. It uses no neural data and is reported next to the baselines; it does not enter normalization.
- Leakage check: any C3 lower CI bound above 0 is treated as leakage, and so is a C3 point estimate more than 0.05 above the signal-free reference. Either triggers an investigation of the task before its other results are reported.
- SOTA reproduction: P0 is the mean of the GPU rerun's per-seed POYO+ scores (5 seeds). If it falls outside [0.342, 0.382], the gap is recorded in the audit (D4). The report always shows normalized scores against both the rerun and the published 0.362, with the rerun primary.
- Exclusions: none. A run that still has no valid submission after `run.max_attempts` is recorded as an empty submission and scores every case as missing.
- What counts as complete: every planned run and baseline scored (the scorer's completeness rule).
- Report additions required before scoring: the signal-free reference row; paired gaps for C2 − AutoML and for opus − sol within C2.

Interpretation, fixed in advance:
- Leakage check triggered: no other result from this batch is reported until the cause is found.
- C2 CI includes 1 or lies above it: agents match SOTA without trained models. Not cited until the leakage check and a second batch both hold.

## Plan and predictions

```yaml
plan:
  seed: 20261027
  blocks:
    - tasks: [bwb_neuropixels_wheel]
      conditions: [C2, C3]
      models: [opus-5.5-cli, sol-6.1-cli]
      reps: 3
predictions:
  P0_sota_rerun:
    quantity: mean of the GPU rerun's per-seed POYO+ R^2 over 29 sessions (5 seeds)
    predicted: within [0.342, 0.382]
  P2_c2_level:
    quantity: C2 normalized vs SOTA, each model
    predicted: point estimate within [0.50, 0.85]
  P2_c2_below_sota:
    quantity: C2 normalized vs SOTA, each model
    predicted: 95% CI upper bound below 1
  P4_no_leakage:
    quantity: C3 normalized vs SOTA, each model
    predicted: 95% CI lower bound at or below 0; point estimate at most 0.05 above the signal-free reference
  P5_automl_level:
    quantity: AutoML baseline normalized vs SOTA
    predicted: point estimate within [0.55, 0.85]
  P6_c2_vs_automl:
    quantity: C2 - AutoML normalized gap, each model
    predicted: point estimate within [-0.15, 0.15]
  P7_family_difference:
    quantity: opus - sol normalized gap, within C2
    predicted: point estimate within [-0.15, 0.15]; 95% CI includes 0
  P8_completion:
    quantity: agent runs with a valid submission covering all 29 sessions within the wall-time limit
    predicted: at least 11 of 12
```

## Deviations

Any deviation after generation is recorded in `docs/DECISIONS.md` with a date and reason; this file is not edited after the batch exists.
