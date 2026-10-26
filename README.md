# NeuroAgentBench

Can general LLM agents, given tools and scientific knowledge, analyse neuroscience data as well as or better than task-specific trained models, and how much of their performance comes from using those trained models as tools?

- Plan and deliverables: `docs/PROJECT.md`
- Scoring, conditions and integrity: `docs/DESIGN.md`
- Task-selection rule and registry: `docs/TASK_SELECTION.md`
- How to run: `docs/RUNBOOK.md`

Each task is a spec in `tasks/<id>/spec.yaml`. Agents run in a sandboxed workspace that holds only the dev data, the test inputs and the condition's tools; a separate scorer compares their predictions with test labels the agent cannot read.
