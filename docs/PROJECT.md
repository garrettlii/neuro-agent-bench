# NeuroAgentBench: project plan

## Question

Can general LLM agents, given tools and scientific knowledge, analyse neuroscience data as well as or better than task-specific trained models, and how much of their performance comes from using those trained models as tools?

## Deliverables and acceptance criteria

| ID | Deliverable | Accepted when |
|---|---|---|
| D1 | Repository and integrity tooling | Tests pass; a dummy task runs end to end; a test proves the agent cannot read the scorer's files. |
| D2 | Task-selection rule and registry | Every candidate considered has a recorded decision and reason in `docs/TASK_SELECTION.md`. |
| D3 | Pilot suite | The five existing tasks are audited and re-specified; NiCLIP and NSD audit reports are written; specs are frozen; the pilot is pre-registered and run. |
| D4 | Baselines per task | Simple baseline, SOTA rerun on the same split, and AutoML with the same budget; each SOTA rerun reproduces its published number or the gap is documented. |
| D5 | Agent conditions harness | Program mode with all tools (C1), generic tools only (C2) and scrambled signal (C3); investigate mode (C4) and native no-code mode (C5) on subsets; at least two model families plus an open-weights model; budgets enforced and logged. |
| D6 | Main suite | 10–15 tasks chosen by the D2 rule, private held-out splits where needed, covering fMRI, EEG/MEG, spikes, calcium imaging, behaviour and BCI, with contamination notes per task. |
| D7 | Results | Normalized scores with CIs, costs, and gaps between conditions; every attempted task reported. |
| D8 | Public release | Someone outside the project can reproduce one result from the docs alone; the private split is kept back. |
| D9 | Stretch: post-training pilot | An open-weights model is fine-tuned on agent traces from dev splits and tested on held-out tasks. |

## Phases

Phases end on exit criteria, not dates.

| Phase | Deliverables | Exit criterion |
|---|---|---|
| Foundation | D1, D2 | D1 and D2 accepted. |
| Pilot | D3, D4 | Pilot specs frozen, pilot pre-registered and run, baselines for every pilot task accepted. |
| Expansion | D5, D6 | All five conditions run on the dummy task and a pilot task; main-suite specs frozen. |
| Results and release | D7, D8 | Final (non-interim) reports for every attempted task; external reproduction of one result. |
| Post-training | D9 | Fine-tuned model scored on held-out tasks under the same harness. |

## Status

- D1: harness, sandbox, guard, scorer and dummy task in place; `pytest -q` passes.
- D2: rule adopted (D-001); registry holds the five pilot tasks only.
- D3: spec stubs and audit checklists created; all fields FILL.
