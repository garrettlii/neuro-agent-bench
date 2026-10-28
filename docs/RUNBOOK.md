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
   For bwb_neuropixels_wheel, the SOTA baseline runs on a separate GPU machine first (D-010): `python scripts/sota_gpu.py export DIR --dry-run`, then `export DIR`, transfer DIR, `run_sota.py DIR --smoke`, one `run_sota.py DIR PART --seeds ...` per GPU or process and `run_sota.py DIR OUT --merge PART...` there (the whole job within 4 h, D-019; instructions at the top of `scripts/sota_gpu_job.py`), transfer OUT back, then `python scripts/sota_gpu.py import OUT`. Until the import is in place, each pass skips the SOTA baseline and its seed references, the agent runs go ahead, and the report stays INTERIM; the next `run_batch.py` pass after the import fills them in.
   Then run `python scripts/probe_network.py --model <name>` once per CLI version and command: `make_batch.py` refuses a pre-registered batch for a model whose latest probe (in `runs_root/network_probes.jsonl`) did not find agent commands offline.
5. `scripts/run_until_done.sh <id>` (or `python scripts/run_batch.py --batch <id> --until-done`, then step 6). Runs that end in an error are retried every `run.retry_interval_s`, up to `run.max_attempts`; after that the run gets an empty submission and scores every case as missing. Without `--until-done`, one pass per invocation. A run that fails within seconds is usually sign-in or quota.
6. `python scripts/score_batch.py --batch <id>`. The report is INTERIM until every planned run and baseline has a scored submission.

## Before any run

- The runner refuses if an agent instruction file (names in `INSTRUCTION_FILES`, `src/nabench/backends.py`) exists in the workspace or any parent directory. Keep `runs_root` outside the home directory's project folders.
- Agent CLIs run as a separate macOS user. Once: `sudo scripts/setup_agent_user.sh` in a terminal, then `python scripts/agent_login.py --model <name>` for each CLI (also in a terminal; sign in again the same way whenever a run fails on authentication). The runner refuses an agent-user run if that user can enter the harness user's home or any data, runs or truth root, and allows one agent-user run at a time.
- Every agent-user run starts with an empty home and temp directory (`sandbox.agent_home`), so CLI state directories (`CLAUDE_CONFIG_DIR`, `CODEX_HOME`) hold no instructions, memory or sessions from earlier runs; only the sign-in files in each model's `keep` are carried over. After each run all of that user's processes are killed and what it wrote outside the workspace is deleted.
