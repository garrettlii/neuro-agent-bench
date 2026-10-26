# Runbook

Data, runs and truth stay outside the repository (`config/default.yaml`). Agent runs need macOS (`sandbox-exec`); on other platforms the runner refuses to start an agent.

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
```

## Dummy task, end to end

1. `python scripts/build_dummy.py` writes the synthetic task to `data_root/tasks/dummy_spikes/` and its labels to `truth_root/tasks/dummy_spikes/`.
2. `python scripts/make_batch.py --smoke` creates a smoke batch (C1, C2, C3 with the scripted stand-in agent). Smoke batches are allowed only for synthetic tasks.
3. `python scripts/run_batch.py --batch <id>` runs the three baselines, then each planned run in a fresh sandboxed workspace.
4. `python scripts/score_batch.py --batch <id>` writes `reports/<id>.md`. Expected: C1 above 1, C2 between 0.5 and C1, C3 with a CI that includes 0.

## Real tasks

1. Fill every FILL in `tasks/<id>/spec.yaml` (`pytest -q` validates all specs) and finish the task's audit in `audits/`.
2. Copy `prereg/TEMPLATE.md`, fill it (plan, predictions, deviations policy), commit and push it together with `config/` and `tasks/`.
3. `python scripts/make_batch.py --prereg prereg/<file>.md`. It refuses unless the file is filled, committed and pushed, `config/` and `tasks/` are clean, every planned spec has no FILL fields, and no earlier batch used the same pre-registration commit (`--additional-batch-reason` records an exception in the ledger).
4. Before the first run of each CLI, confirm its flags with its `--help` output and run it on the dummy task only.
5. `python scripts/run_batch.py --batch <id>`. Runs that end in an error are retried on the next invocation, up to `run.max_attempts`. A run that fails within seconds is usually sign-in or quota.
6. `python scripts/score_batch.py --batch <id>`. The report is INTERIM until every planned run and baseline has a scored submission.

## Before any run

- The runner refuses if an agent instruction file (names in `INSTRUCTION_FILES`, `src/nabench/backends.py`) exists in the workspace or any parent directory. Keep `runs_root` outside the home directory's project folders.
- CLI state directories (`CLAUDE_CONFIG_DIR`, `CODEX_HOME`) point to separate benchmark-only directories with no project instructions or memory.
