# Informal analysis: opus-5.5-cli vs sol-6.1-cli on bwb_neuropixels_wheel

Batch b-20261028-044630-f61b, 2026-10-29. Written at the owner's request, setting aside the standing rules (rule 7
and the D-020 hold). This is exploratory and not citable: the leakage gate (D-020) is triggered, and the paper's
POYO+ number stands in for the GPU rerun the pre-registration made primary.

**Conflict of interest.** The interpretation below was written under the conflict of interest that rule 7 of
`CLAUDE.md` describes, and should be read with that in mind. The numbers come from code
(`runs_root/b-20261028-044630-f61b/investigation/scripts/analysis_published.py`, output `analysis_published.json`
next to it).

**Anchors.** simple = 0.219 and SOTA (POYO+) = 0.362, both from the paper; normalized = (R² − 0.219) / 0.143. Our
train-only refit of the simple baseline, scored per seed like the paper, gives 0.221, so the paper's numbers
match this data version for the linear model. The denominator is small: **0.01 raw R² = 0.07 normalized.**
95% CIs come from the scorer's paired bootstrap over the 29 sessions (2000 draws, seed 0), with reps averaged per
session.

## Results

| row | raw R² | normalized [95% CI] |
|---|---|---|
| simple (paper) | 0.219 | 0 |
| simple, batch's own 5-seed mean of predictions | 0.270 | (0.36) |
| signal-free reference (each session's mean dev curve) | 0.193 | −0.18 [−0.50, 0.12] |
| AutoML (4 h search, refit on train+val) | 0.319 | 0.70 [0.37, 1.01] |
| SOTA POYO+ (paper) | 0.362 | 1 |
| **C2 opus** (3 reps) | **0.428** | **1.46 [1.17, 1.77]** |
| **C2 sol** (3 reps) | **0.419** | **1.40 [1.08, 1.69]** |
| C3 opus (spikes scrambled) | 0.207 | −0.08 [−0.40, 0.21] |
| C3 sol (spikes scrambled) | 0.209 | −0.07 [−0.37, 0.21] |

Per rep, raw R² (normalized):

- C2 opus: 0.433 (1.50), 0.425 (1.44), 0.428 (1.46)
- C2 sol: 0.429 (1.47), 0.393 (1.22), 0.434 (1.50)
- C3 opus: 0.207, 0.207, 0.208; C3 sol: 0.211, 0.206, 0.211

All 12 runs submitted all 29 sessions.

Paired gaps (normalized [95% CI]):

| gap | opus | sol |
|---|---|---|
| C2 − AutoML | 0.76 [0.60, 0.94] | 0.70 [0.54, 0.85] |
| C2 − signal-free reference | 1.65 [1.40, 1.90] | 1.58 [1.34, 1.82] |
| C2 − C3 | 1.55 [1.32, 1.78] | 1.46 [1.25, 1.67] |
| C3 − signal-free reference (leakage criterion 2, margin 0.05) | 0.10 [0.05, 0.16] | 0.12 [0.06, 0.19] |

Opus − sol:

- **C2: 0.068 [−0.042, 0.169]**, i.e. 0.010 raw R².
  - By session, opus is ahead in 18 of 29 sessions and sol in 11. A sign test gives p = 0.27, two-sided.
  - The two models' scores correlate at r = 0.94 across sessions.
  - Between reps, a session's score varies by SD 0.030 for opus and 0.027 for sol.
- **C3: −0.015 [−0.041, 0.005].**

Sessions where C2 beats AutoML: 29/29 for both models. Sessions where C2 beats the signal-free reference: 29/29
for opus, 28/29 for sol.

Costs, mean per run (tokens recounted per D-021; Sol's dollar cost is not reported by its CLI):

| | tokens in / out | wall time | USD (reported by the CLI) |
|---|---|---|---|
| C2 opus | 1.83 M / 15.7 k | 52 min | 1.08 |
| C2 sol | 2.53 M / 34.5 k | 24 min | n/a |
| C3 opus | 0.78 M / 12.5 k | 12 min | 0.70 |
| C3 sol | 1.11 M / 24.2 k | 15 min | n/a |

Re-running the saved C2 programs (2b) on scrambled spikes, normalized (difference from the signal-free reference):

- opus r1: −0.58 (−0.39)
- sol r1: −0.17 (+0.02)
- sol r2: −0.14 (+0.04)
- sol r3: −0.22 (−0.03)

Anchor sensitivity (point estimates):

| simple anchor | AutoML | C2 opus | C2 sol | C3 opus | C3 sol |
|---|---|---|---|---|---|
| paper, 0.219 | 0.70 | 1.46 | 1.40 | −0.08 | −0.07 |
| our per-seed refit, 0.221 | 0.70 | 1.47 | 1.40 | −0.10 | −0.09 |
| batch's 5-seed mean, 0.270 | 0.54 | 1.72 | 1.61 | −0.68 | −0.65 |

The last row mixes an averaged-predictions simple with a single-model SOTA, so it is not like for like.

### Pre-registered predictions, against the paper's anchors

| prediction | result | outcome |
|---|---|---|
| P0 SOTA rerun in [0.342, 0.382] | no rerun | not testable |
| P2 C2 level in [0.50, 0.85] | 1.46 / 1.40 | **fails** (above) |
| P2 C2 upper CI below 1 | lower CIs 1.17 / 1.08 | **fails** (above SOTA) |
| P4 no leakage: C3 lower CI ≤ 0 | −0.40 / −0.37 | holds |
| P4 no leakage: C3 − reference ≤ 0.05 | 0.10 / 0.12 | **fails** |
| P5 AutoML in [0.55, 0.85] | 0.70 | holds |
| P6 C2 − AutoML in [−0.15, 0.15] | 0.76 / 0.70 | **fails** (above) |
| P7 opus − sol in [−0.15, 0.15], CI includes 0 | 0.07 [−0.04, 0.17] | holds |
| P8 ≥ 11/12 complete | 12/12 | holds |

## Interpretation (informal)

### 1. The leakage trigger looks like a reference problem, not leakage

Criterion 2 trips because the C3 runs beat the signal-free reference by about 0.015 raw R². That is 0.10 to 0.12
normalized, against a margin of 0.05 normalized (0.007 raw). Four facts point to the reference being too weak,
not to spikes leaking information:

- The three opus C3 programs read no spikes at all. They use a per-session mean curve that weights recent trials
  more, and they still clear the margin (0.10 [0.05, 0.16]).
- The reference is the plain, unweighted mean of each session's dev trials. The split is causal (40/20/40 in
  time), so test trials come after every dev trial. Weighting recent trials picks up within-session drift
  (fatigue, satiety, engagement) that the plain mean misses. All 6 C3 programs used recency weighting.
- The C2 programs, re-run on scrambled spikes, land at or below the reference (+0.04 at most). This holds even for
  the most complex ones, so they do not find structure in scrambled data.
- Criterion 1, which asks whether the C3 lower CI is above 0, holds for both models.

On this reading, the C3 agents did the right thing: they noticed the spikes carried no signal and fell back to the
best label-only predictor they could build. The gate fired because the pre-registered reference is easier to beat
than "the stereotyped time course alone".

### 2. "Agents beat POYO+" is probably not robust

C2 sits at 1.40 to 1.46, with both lower CIs above 1. Taken at face value, generic agents beat a model pretrained
on 423 sessions by about 0.06 raw R². Three asymmetries in this setup favour the agents, so the headline should not be
taken as stated:

- **More, and more recent, training data.** All 12 agent programs refit on train+val, which is 45% more trials,
  and the val block is the stretch closest in time to test. POYO+ and simple trained on train only and kept the
  best epoch on val. Given the drift above, recent trials are worth more than their count suggests. AutoML also
  refits on train+val and still lands at 0.319, so this asymmetry alone does not produce 0.42. How much it is worth
  for POYO+ is what the D-022 variant would measure.
- **Single model vs ensemble.** The paper's 0.362 is the mean of single-seed scores. The agents' final predictions
  are ensembles or blends: opus r1 averages 5 CNN seeds and blends in ridge, opus r2 averages 3 networks with
  ridge, and the sol programs mix several model families. Averaging alone lifts the simple baseline from 0.221 to
  0.270 (+0.05). If POYO+ gained anything similar, its ensemble would sit near 0.41, roughly level with the agents.
  That is why the pre-registration made the 5-seed GPU rerun primary.
- **Data version.** The paper's POYO+ ran on v0.0.9, and we have v0.1.0. Trial counts match, and the linear model
  reproduces within 0.002, but that does not show POYO+ would.

What does look robust:

- Both agents beat a 4-hour AutoML search by about 0.7 normalized (0.10 raw), in every one of the 29 sessions,
  even though AutoML also refit on train+val.
- The C2 − C3 drop (about 1.5 normalized) shows the C2 scores come mostly from neural signal, not from the label
  template.

A fair summary: the agents reach at least POYO+'s neighbourhood without any pretrained model. Whether they exceed
it needs the ensemble rerun, and ideally the train+val variant.

### 3. Opus vs sol: no detectable difference in score; a clear difference in reproducibility

- **Score.** C2 differs by 0.068 normalized (0.010 raw), with a CI spanning 0. Opus leads in 18 of 29 sessions
  (p = 0.27), and the gap is smaller than the spread between reps. Most of opus's lead comes from one weaker sol
  rep (0.393). Sol's other two reps match or beat opus's best. Per-session scores correlate at 0.94, so both
  models find the same sessions easy and hard. P7 (indistinguishable) holds. In C3 the two models are level.
- **Consistency.** Opus's three C2 reps sit within 0.009 R² of each other. Sol's span 0.040 because of one rep.
  With three reps this is a hint, not a finding.
- **Reproducibility** is where they clearly differ:
  - Sol: 2 of 3 C2 programs re-ran bit for bit from the saved code. The third (r1) differs by less than 1e-6
    in R².
  - Opus: only 1 of 3 C2 programs could be re-run at all, and it did not reproduce exactly (0.435 vs 0.433). The
    other two ended with a blend step that was never saved, run with settings passed through environment
    variables that were not recorded. C3 opus r3 left no code at all.
  - If the benchmark counts an auditable pipeline as part of "analysing data as well as" a trained model, opus
    loses there on this evidence.
- **Style.**
  - In C2, opus built neural-network-plus-ridge blends on binned counts. Sol built multi-model ensembles,
    with the weights chosen on val.
  - In C3, opus dropped the spikes entirely. Sol kept a template-plus-neural-residual design (predicting the
    deviation from each session's mean curve) with a down-weighted neural term. Some sol C3 programs also used
    cross-session pooling, trend extrapolation and the number of test trials.
  - The two designs reach the same place in both conditions.
- **Cost.**
  - Sol used about 40% more input tokens and twice the output tokens of opus.
  - Sol finished C2 in half the wall time (24 vs 52 min). Opus r1's time includes about 25 minutes of CNN
    training.
  - Opus cost about $1.08 per C2 run by its own CLI's count; Sol's cost isn't reported.

### 4. What to check next

- The GPU rerun with the 5-seed ensemble, which settles item 2. The train+val variant would show how much refitting
  on more recent data is worth for POYO+.
- A recency-weighted label-only reference, alongside the pre-registered one. This is a design choice for you, and
  it belongs in a new pre-registration, not in this batch.
- More reps, or a second task, before reading anything into opus vs sol on score.
