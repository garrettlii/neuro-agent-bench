# Decision log

Append-only. Any change to a frozen spec, metric, baseline, budget or scoring rule after a batch exists needs a dated entry here and a new pre-registration.

## 2026-10-26

- **D-001 Project created; selection rule adopted.** NeuroAgentBench asks whether general LLM agents with tools and scientific knowledge analyse neuroscience data as well as task-specific trained models, and how much of their performance comes from using those models as tools. The task-selection rule in `docs/TASK_SELECTION.md` (hidden or post-cutoff test labels or a possible private re-split; runnable published SOTA; at most about one GPU-day per run; added modality coverage) is adopted before any candidate is assessed. The five pilot tasks predate the rule and are reported separately. Integrity tooling is adapted from dark-gene-bench: pre-registration guard (now also requiring a clean `tasks/`, and carrying the run plan), batch ledger with one batch per pre-registration commit, data, run and truth roots outside the repository, completeness rule for final reports, and CLI backends with empty stdin and recorded commands and versions. The per-model empty working directory becomes a fresh workspace per run attempt, inside an OS sandbox.

- **D-002 Name to be reconsidered before public release (collision with NeuroAgent).** NeuroAgent (arXiv 2605.06584) is a neuroimaging agent framework with a near-identical name. Decide on the name before D8 (public release); see `docs/RELATED_WORK.md`.
