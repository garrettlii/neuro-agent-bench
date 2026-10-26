# Pre-registration

Every non-synthetic batch is generated from a pre-registration file in this folder. The batch generator checks that the file

1. has no unfilled placeholders,
2. is committed with no local changes, with `config/` and `tasks/` also clean (task specs are frozen with it), and
3. is on a remote branch (pushed), which gives an external timestamp.

The run plan is read from the file's YAML block, so the plan is frozen too. The generator records the commit hash, plan and predictions in the batch manifest and appends the batch to `truth_root/ledger.jsonl`. A second batch under the same pre-registration commit is refused unless `--additional-batch-reason` is given, and the reason is logged.

## Workflow

1. Copy `TEMPLATE.md` to a new file named after the batch purpose (for example `pilot.md`).
2. Fill in every FILL field, including the plan and the predictions in the YAML block.
3. Commit and push it with `config/` and `tasks/`. Only then generate the batch.
4. After scoring, do not edit the file. Record deviations in `docs/DECISIONS.md` and write a new pre-registration for any new batch.

Thresholds and predictions are set by the human authors, not by any evaluated model.
