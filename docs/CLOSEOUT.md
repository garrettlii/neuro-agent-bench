# Close-out

The project closed on 2026-11-07 (D-024). This file contains aggregates only. Per-case results, transcripts, workspaces and truth stay under `data_root`, `runs_root` and `truth_root`.

**Every number from pilot batch b-20261028-044630-f61b is exploratory.** The batch's leakage gate triggered (D-020). Under the pre-registration, that means no result from the batch is reported, so none of these numbers can be cited. They come from the owner-approved informal analysis (`reports/b-20261028-044630-f61b-informal-analysis.md`) and from the investigation outputs in `runs_root/b-20261028-044630-f61b/investigation/`. Unless stated otherwise, normalized scores use the paper's anchors (simple 0.219, POYO+ 0.362, so 0.01 raw R² = 0.07 normalized), not the GPU rerun the pre-registration made primary. That rerun was never run (D-023).

**Conflict of interest.** The owner set aside rule 7 of `CLAUDE.md` for this file (D-024). Sections 3 to 5 include interpretation, including of the Claude results. Section 7 makes recommendations formed after seeing the results. Both were written under the conflict of interest that rule 7 describes, and should be read with that in mind.

**Bottom line** (exploratory; interpretation):
- The harness and integrity tooling work end to end.
- On the one task that was run, both agents, working without trained models:
  - scored at or above the paper's POYO+ number;
  - scored about 0.7 normalized above a 4 h AutoML search.
- Two asymmetries favour the agents: they refit on train+val, and their ensembles were compared with a single-model anchor. The 5-seed SOTA rerun that would have settled this was never run.
- The leakage gate fired, most likely because its reference was weaker than simple label-only predictors.
- Opus and Sol could not be told apart on score. They did differ on how reproducible their programs were.

## 1. What was built

- **Harness** (`src/nabench/`):
  - Runner: a fresh workspace per attempt, a hard wall-time kill, and retries up to `run.max_attempts`, followed by an empty submission.
  - CLI backends: empty stdin, with commands, versions and token use recorded (D-005, D-021).
  - A scripted stand-in agent and the dummy task.
  - Batch manifests holding the SHA-256 of every staged input and spec.
  - Scorer: normalized scores, a paired bootstrap over cases, a column against the published SOTA (D-007), the completeness rule with INTERIM marking, the leakage check, and the C3 scramble controls.
- **Integrity tooling:**
  - Pre-registration guard: the pre-registration must be filled, committed and pushed, `config/` and `tasks/` must be clean, and each pre-registration commit gets one batch, recorded in the ledger.
  - The data, runs and truth roots are kept outside the repository, enforced at config load.
  - The runner refuses to start if an instruction file is reachable from the workspace.
  - The stand-in runs under `sandbox-exec`. The agent CLIs run as a separate macOS user. Before every run the harness checks that this user cannot enter protected paths. Each run gets an empty home, and anything the run left behind is cleaned up afterwards (D-004).
  - A network probe runs for each CLI version and command (D-003).
- **Tests:** 88 tests in total, on synthetic data with no network access. 86 pass and 2 skip, because the venv has no torch. They include the D1 test that the agent cannot read the scorer's files.
- **BWB task (`bwb_neuropixels_wheel`):**
  - Built from the benchmark's processed v0.1.0 evaluation sessions: 29 sessions, every sample checked against the benchmark's own loader. Spec, audit and data notes are in place.
  - Baselines: simple (published Linear configurations, D-012), AutoML (D-013) and a signal-free profile reference.
  - C3 circular-shift scramble (D-014).
  - POYO+ SOTA wrapper with a CPU attention path (D-009), plus a GPU export, merge and import path with manifests and a 4 h rule (D-010, D-018, D-019). An exploratory train+val variant of that job exists but was never run (D-022).
- **Docs:** design, plan, task-selection rule and registry, data, runbook, related work, and one audit checklist per pilot task. Only the BWB audit is substantially filled.

## 2. The pilot batch and its predictions

| | |
|---|---|
| Batch | b-20261028-044630-f61b |
| Pre-registration | `prereg/pilot-bwb.md` at eb0ccf4 |
| Task | bwb_neuropixels_wheel |
| Conditions | C2, C3 |
| Models | opus-5.5-cli, sol-6.1-cli |
| Reps | 3 (12 agent runs) |

Scored: simple, AutoML, the profile reference and all 12 agent runs. The SOTA baseline and its 5 seed references never ran. The official report (`reports/b-20261028-044630-f61b.md`) is therefore INTERIM: 6 of 21 runs and baselines are unscored, and it has no normalized scores. It will stay that way (D-023).

Deviations recorded after generation: D-016 to D-019, D-021 and D-022.

Pre-registered predictions, scored against the paper's anchors (exploratory):

| prediction | result | outcome |
|---|---|---|
| P0 SOTA rerun in [0.342, 0.382] | no rerun | not testable |
| P2 C2 level in [0.50, 0.85] | opus 1.46, sol 1.40 | failed (above) |
| P2 C2 upper CI below 1 | lower CIs 1.17 / 1.08 | failed (above) |
| P4 C3 lower CI ≤ 0 | −0.40 / −0.37 | held |
| P4 C3 − reference ≤ 0.05 | 0.10 / 0.12 | failed |
| P5 AutoML in [0.55, 0.85] | 0.70 | held |
| P6 C2 − AutoML in [−0.15, 0.15] | 0.76 / 0.70 | failed (above) |
| P7 opus − sol in [−0.15, 0.15], CI includes 0 | 0.068 [−0.042, 0.169] | held |
| P8 at least 11 of 12 runs complete | 12 / 12 | held |

Four held, four failed and one could not be tested. H0 (SOTA reproduces) was never tested.

## 3. The leakage gate and the investigation

**Trigger** (exploratory):
- Criterion 2 triggered for both models. C3 minus the signal-free reference is 0.10 [0.05, 0.16] for opus and 0.12 [0.06, 0.19] for sol, against a margin of 0.05 normalized, which is 0.007 raw R².
- Criterion 1 did not trigger: the C3 lower CI bounds are −0.40 and −0.37.
- D-020 judged the trigger against the published SOTA and made it final only once a SOTA rerun was scored. With no rerun (D-023), it stays provisional.
- Following the pre-registration, the batch's results are not reported. The owner nevertheless published the interim report and the informal analysis, neither of them citable (D-020).

**Investigation** (read-only on the batch; all numbers exploratory):

- **Access (2d).** Across the 12 runs there were 0 network attempts, 0 reads outside the workspace and 0 accesses to truth or score paths. The pattern scan flagged 26 hits, and all were checked by hand and found to be false positives. Limit: the opus runs keep no transcript, so for them only refused tool calls and the code left in the workspace could be scanned. C3 opus r3 left no code.
- **C3 programs (2c).**
  - Inputs used across the 6 runs:
    - 6/6 fit their final model on train+val (one by self-report only).
    - 6/6 weight recent trials more.
    - 6/6 use per-session statistics of the target.
    - 3/6 use spike counts or rates.
    - 3/6 use no spikes at all (one by self-report).
    - 2/6 extrapolate a trend to the test horizon.
    - 2/6 use the number of test trials.
    - 3/6 pool label curves across sessions (one by self-report).
  - The pre-registered reference is the unweighted mean of each session's dev trials.
  - The three C3 programs that use no spikes still exceed the margin, at 0.10 [0.05, 0.16].
- **C2 programs re-executed (2b).**
  - Re-runs as saved: 4 of 6 programs could be re-run.
    - Sol: 3/3 re-ran. Two reproduced bit for bit and one within 1e-6 R².
    - Opus: 1/3 re-ran, scoring 0.435 against 0.433 submitted.
    - The other two opus programs depend on a final blend step that was never saved, run with settings that were not recorded.
  - On C3-scrambled inputs, the 4 re-runnable programs scored −0.58, −0.17, −0.14 and −0.22 normalized. That is at most 0.04 above the reference.
- **Data version (2e).** The published Linear and POYO+ numbers ran on internal builds v0.0.11 and v0.0.9, and neither build is public. Unit and train, val and test counts match v0.1.0 in all 29 sessions. Whether each split holds the same trials, and whether wheel processing changed, cannot be determined.
- **Tokens (D-021).** The parser now uses Claude's cumulative `modelUsage`.
  - Recorded C2 opus input was 292,350 tokens; the recomputed cumulative input is 5,492,650.
  - The largest run used 2.72 M tokens in and out, under the 3 M budget.
  - Sol's totals cannot be checked against session logs, because none exist.
  - The resumed-run case has a test (`tests/test_backends.py`).

**Interpretation.** The trigger most likely reflects a weak reference, not leakage:
- The three C3 programs that read no spikes still clear the margin.
- All six C3 programs weight recent trials more. The split is causal, so test comes after every dev trial. A recency-weighted template therefore picks up within-session drift that the unweighted reference misses.
- The C2 programs, re-run on scrambled spikes, land at most 0.04 above the reference, so they find no structure in scrambled data.
- Criterion 1 holds for both models.
- On this reading, the C3 agents saw that the spikes carried no signal and fell back to the best label-only predictor they could build.

No pre-registered check tests this, so the batch stays uncitable.

## 4. Simple-baseline inflation and the train/val finding (exploratory)

- **Simple baseline.**
  - The batch's simple baseline averages the 5 eval seeds' predictions and scores 0.270 raw R².
  - The benchmark instead scores each seed and then averages the scores. A train-only refit here, scored that way, gives 0.221 against the published 0.219.
  - The gap is 0.049 raw, about 0.35 normalized.
  - The batch's SOTA baseline was also defined as a 5-seed ensemble of predictions (D-006, D-010), while P0 checks the mean of the per-seed scores.
  - With the batch's 0.270 as the 0 anchor, the point estimates become: AutoML 0.54, C2 opus 1.72, C2 sol 1.61. This mixes an ensembled simple with a single-model SOTA.
- **Train/val.**
  - Every agent program fit its final model on train+val: 12/12, with C3 opus r3 by self-report only. So did AutoML, by design.
  - Simple and SOTA train on train only and use val to pick the best epoch (`2a_fit_split.md`).
  - Train+val holds 1.445 times as many trials as train. The split is causal, so val is the block closest in time to test.
  - AutoML, also refit on train+val, scored 0.319 raw (0.70).
  - The D-022 variant, which would have measured what a train+val refit is worth for POYO+, will not run (D-023).

**Interpretation.** Both asymmetries favour the agents over POYO+ as the paper reports it:
- **More training data.** The agents trained on 45% more trials, including the block closest in time to test.
- **Ensembles against a single model.** They submitted ensembles or blends, which were scored against a single-model number. Averaging alone lifts the simple baseline by 0.05 raw. If POYO+ gained as much, its ensemble would sit near 0.41, about level with the agents (0.42 to 0.43).
- **The refit alone does not explain the scores.** AutoML, also refit on train+val, still lands at 0.319.

The claim "agents beat POYO+" is therefore probably not robust. Generic agents reach at least POYO+'s neighbourhood without a pretrained model. Whether they exceed it is what the 5-seed rerun and the D-022 variant would have shown.

What does look robust:
- Both agents beat a 4 h AutoML search by about 0.7 normalized (0.10 raw) in every one of the 29 sessions.
- The C2 − C3 drop, about 1.5 normalized, shows that the C2 scores come mostly from neural signal.

## 5. Opus and Sol on C2 (exploratory)

The two are not distinguishable on C2:

| | value |
|---|---|
| opus − sol | 0.068 [−0.042, 0.169] normalized (0.010 raw R²) |
| mean raw R², opus / sol | 0.428 / 0.419 |
| sessions where opus is ahead | 18 of 29 (sign test p = 0.27) |
| opus − sol, C3 | −0.015 [−0.041, 0.005] |

P7 held.

**Interpretation:**
- **Score: no detectable difference.**
  - The gap (0.010 raw) is smaller than the spread between reps.
  - Most of opus's lead comes from one weaker sol rep (0.393). Sol's other two reps match or beat opus's best.
  - Per-session scores correlate at r = 0.94, so both models find the same sessions easy and hard.
  - In C3 the two are level.
- **Consistency.** Opus's three C2 reps sit within 0.008 raw R² of each other. Sol's span about 0.04, because of that one rep. With three reps this is a hint, not a finding.
- **Reproducibility is where they clearly differ.**
  - All 3 sol C2 programs re-ran from the saved code; 1 of 3 opus programs did.
  - One opus C3 run left no code.
  - If an auditable pipeline counts as part of analysing data as well as a trained model, opus is behind on this evidence.
- **Cost.** Per C2 run, sol used about 40% more input tokens (2.53 M vs 1.83 M) and over twice the output tokens (34.5 k vs 15.7 k), and finished in under half the wall time (24 vs 52 min). Opus cost about $1.08 per C2 run by its CLI's count. Sol's CLI reports no dollar cost.

## 6. What remains undone

- **C1** was deferred to a Linux GPU harness (D-011). It was never run.
- **SOTA rerun:** never run, and no GPU work will be done (D-023). As a result:
  - H0 and P0 are untested.
  - D4 is incomplete for BWB.
  - The official batch report stays INTERIM.
  - The BWB audit's rerun and gap fields stay FILL.
- **The other four pilot tasks** (niclip_map_to_paper, nsd_cross_subject_id, stop_signal_behaviour, t15_speech) have specs and audits with FILL fields, and no data, baselines or runs. D3 is not accepted.
- **D5:** only C2 and C3 ran on a real task, with two model families. C1, C4 and C5 have not been run on a real task. There is no open-weights model and no enforced cost cap (D-003).
- **D6 to D9:** not started. The name (D-002) and the request to IBL about redistribution (D-008) are still open.
- **A citable result** would need a new pre-registered batch.

## 7. Recommendations for future benchmarks

These were formed after seeing the results. They are recommendations, not adopted decisions: a future benchmark adopts them in its own pre-registration. None of them changes this batch.

1. **Measure the SOTA rerun on the harness's own hardware before admitting a task.** Run the published SOTA for one session and one seed, and project the full time, before rule (c) is assessed. Here, POYO+ could not run on MPS, and one seed on CPU was projected at about 8.5 times the limit (D-009). The GPU path that followed took four more decisions (D-010, D-018, D-019, D-022) and never ran.
2. **Make the primary anchors match how the published numbers are computed.**
   - Score simple and SOTA per seed and then average the scores, as the paper does.
   - Report ensembles of predictions as separate rows, and compare agent ensembles with them.
   - Here, the two ways of scoring differ by 0.35 normalized on the simple anchor alone.
3. **Refit every baseline on train+val, with the epoch count or hyperparameters chosen on val** (as the D-022 variant does), and say in TASK.md that final fits may use all dev data. Agents refit on train+val unprompted (12/12), so baselines that train on train only start behind.
4. **Make the leakage gate test the strongest label-only reference, with a CI-based criterion.**
   - Add a recency-weighted per-session mean curve, with its time constant chosen on val, next to the unweighted one, and test C3 against the stronger of the two.
   - Trigger when the lower CI bound of C3 minus that reference is above 0, not when a point estimate exceeds a fixed normalized margin.
   - Here, a 0.05 normalized margin was 0.007 raw R², and three C3 programs that read no spikes cleared it.
5. **Prefer tasks whose SOTA − simple gap is large relative to run-to-run noise, and always report raw differences next to normalized ones.** On BWB the gap is 0.143 raw, so 0.01 raw R² equals 0.07 normalized, and the spread between reps of one model reached 0.04 raw.
6. **Require a re-runnable final program and keep transcripts.**
   - Ask agents for one script, such as `submission/run.sh`, that regenerates the predictions from the staged data. Re-run it at scoring time in a fresh workspace and report whether it reproduces.
   - Keep each CLI's transcript under `runs_root` (no `--no-session-persistence` or `--ephemeral` for scored runs), so runs can be audited and token totals checked.
   - Here, 2 of the 6 C2 programs could not be re-run as saved, and one C3 run left no code.
7. **Do not compare model families on one task with three reps.**
   - Plan at least two tasks, and enough reps that the expected gap exceeds the spread between reps, before pre-registering a family comparison.
   - Record reproducibility and cost per model alongside score, since that is where the two families differed here.
8. **Test token parsers on resumed runs, and on every CLI's real output, before a batch** (D-021).
9. **Confirm the published data version is obtainable when selecting a task** (2e).
10. **Keep scratch outputs under `runs_root`, not system temp.** The scratch copies cited in 2e were deleted by the OS temp cleaner before close-out. The scripts had already been copied into the investigation folder.

## 8. Where things are

- **In the repository:** code, specs, docs, the pre-registration, and the batch's interim and informal reports.
- **Outside the repository** (`config/default.yaml`), none of them deleted:
  - `~/benchmarks/nab/nab-data`, `nab-runs` and `nab-truth`.
  - The GPU exports `~/benchmarks/nab/sota-export` and `sota-export-2`.
  - The agent user's area, `/Users/Shared/nab-agent`.
- **What to keep or delete** is the owner's decision.
