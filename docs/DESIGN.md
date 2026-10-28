# Design

## Score

For each task, per test case, the scorer computes the task metric for the agent, the simple baseline and the SOTA rerun. The normalized score is

    normalized = (agent - simple) / (SOTA - simple)

0 is the simple baseline, 1 matches SOTA, and above 1 beats it. The same formula works for lower-is-better metrics. Where a task has a noise ceiling, the report also gives (agent - simple) / (ceiling - simple).

- Metrics are per-case and aggregate by the mean, so every CI is a paired bootstrap over test cases (the same resampled cases for agent, simple and SOTA). 95% percentile intervals, 2,000 draws by default.
- Repetitions of a run are averaged per case before bootstrapping.
- Across tasks: the mean normalized score, bootstrapping cases independently within each task. Tasks with different `selection` (dummy, pilot chosen before the rule, admitted by the rule) are never pooled.
- Cost (tokens in and out, wall time, GPU-hours) is reported next to every score. Tokens come from each CLI's own output, with input counting cached input too (D-005); GPU-hours are recorded where the run reports them and otherwise shown as n/a.
- Missing or unparseable predictions score as the worst value for that case (for accuracy, 0), and the number of missing cases is reported.
- Baselines run on the same Mac as the agents. If a SOTA rerun cannot finish there within the task's wall time, the task is recorded as failing rule (c) on this harness.

## Conditions

| ID | Mode | Tools | Data | Purpose |
|---|---|---|---|---|
| C1 | program | all (generic + trained models) | real | core condition |
| C2 | program | generic only | real | C1 − C2 measures how much the agent depends on trained models as tools |
| C3 | program | all | scrambled (phase-randomized or label-permuted) | any skill above baseline means leakage |
| C4 | investigate | all | real | subset of tasks; agent documents its exploration before predicting |
| C5 | native (no code) | generic listed, code execution disabled | real | subset of tasks |

The prompt and `TASK.md` are identical across models, and identical between C1 and C3. C2's `TASK.md` differs only in not listing the trained-model tools, which are not staged.

## Integrity

- **Scorer outside the sandbox.** No agent can read or write the repository, `data_root`, `runs_root` (other workspaces, logs and run records) or `truth_root`. The scripted stand-in runs under an OS sandbox (macOS `sandbox-exec`) that denies those paths, writes outside its workspace and temp directories, and network. Agent CLIs sandbox their own commands with `sandbox-exec`, which cannot start inside such a profile, so they run instead as a separate macOS user (`sandbox.agent_user`) that cannot enter the harness user's home or any of those roots (checked before every run). Each workspace is opened to that user by an ACL for its run only. Every run gets an empty home and temp directory, holding only the CLI's sign-in files; when it ends, all of that user's processes are killed and what it wrote outside the workspace is deleted, so nothing carries over to later runs. The CLI reaches its own API, and its own sandbox must keep agent commands offline, which a recorded network probe per CLI version and command checks before any pre-registered batch.
- **No test labels or scores.** Test labels live only under `truth_root` (owner-only permissions) and are never staged. Scores are computed after the run and are never returned to the agent.
- **Limited submissions.** Whatever is at `submission/predictions.csv` when the agent exits or is stopped is copied to the scorer, up to `budgets.submission_limit` per task × condition × model. Links that point outside the workspace are refused.
- **Fixed budgets.** Wall time is a hard limit (the process group is killed); tokens are logged per CLI; the AutoML baseline gets the same wall-time and GPU budget as the agent (checked by the spec validator).
- **Pre-registration.** Every non-synthetic batch needs a filled, committed and pushed pre-registration whose YAML block holds the run plan and predictions; `config/` and `tasks/` must be clean, and every spec in the plan must have no FILL fields. One batch per pre-registration commit unless a reason is recorded in the ledger.
- **Every attempted task reported.** The report lists every task in the batch. It is marked INTERIM until every planned run and baseline has a scored submission. There are no exclusions: a run that still errors after `run.max_attempts` gets an empty submission, so every case scores as missing.
- **Multiple model families, deterministic scoring.** At least two families plus an open-weights model; all scoring is deterministic code.
- **Provenance.** Each batch manifest records the plan, pre-registration commit, config, resolved specs, SHA-256 of every staged input and of each spec, and each model's command and version. Staged files are re-checked against these checksums at every run. Each run record holds the CLI version, exit status, wall time and token use.

## Layout

| Where | What |
|---|---|
| `tasks/<id>/spec.yaml` | task spec, validated against `tasks/schema.json` |
| `src/nabench/` | harness, scorer, metrics, scramble controls, task code |
| `data_root/tasks/<id>/` | dev inputs and labels, test inputs, tool weights |
| `truth_root/tasks/<id>/` | test labels, noise-ceiling predictions |
| `runs_root/<batch>/` | manifest, `runs.jsonl`, per-run workspaces and logs |
| `truth_root/<batch>/` | manifest with truth checksums, submissions, per-case scores |
| `truth_root/ledger.jsonl` | every batch and its pre-registration |
| `reports/<batch>.md` | aggregates only |
