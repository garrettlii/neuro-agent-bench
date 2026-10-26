# Related work

How NeuroAgentBench relates to nearby benchmarks and systems. Descriptions come from each paper's arXiv abstract or HTML as read on 2026-10-26, and mostly from abstracts. Check against the full papers before citing.

## What this project adds

1. **A SOTA rerun and normalization on every task.** The published SOTA method is rerun on the same split, and its result is checked against the published number; any gap is documented. Agent scores are reported as `(agent - simple) / (SOTA - simple)` per test case (`docs/DESIGN.md`). 0 means the simple baseline, and 1 means the trained task-specific model on identical data.
2. **The trained-model-tools ablation, C1 − C2.** In C1 the agent gets all tools, including the task's trained models. In C2 it gets generic packages only. The difference measures how much of the agent's score comes from calling trained models rather than analysing the data itself.
3. **Two controls.** C3 runs the agent on scrambled signal (label-permuted or phase-randomized), so any skill above baseline points to leakage or shortcuts. An AutoML baseline gets the same wall-time and GPU budget as the agent, which separates agent skill from automated search at equal compute.
4. **Modality breadth from a written selection rule.** Tasks span fMRI, EEG/MEG, spikes, calcium imaging, behaviour and BCI. They are admitted by the rule in `docs/TASK_SELECTION.md`, which was adopted before any candidate was assessed (D-001), not hand-picked.

## Agent benchmarks in neuroscience

**BrainPilotBench-v0** (arXiv 2607.15079, released with the BrainPilot agent)
- *Measures:* four brain-science analysis tasks: RSC place-cell analysis (two-photon imaging and VR behaviour), TOPS-fMRI tonic-pain signature, BCI Competition IV 2a motor imagery, and Sleep-EDF staging. Submitted artifacts are graded on a [0, 1] scale, with domain knowledge on and off. Splits, prompts and scoring are hashed before runs. Runs execute in a no-network container with labels held outside the workspace. An oracle/empty-submission gate validates each grader. It compares agent harnesses (BrainPilot, Codex, Claude Code) and backbones, one run per cell. The RSC metric includes improvement over shuffled decoding.
- *Difference:* we found no SOTA rerun on the benchmark splits or normalization against one. It has no trained-model-tools ablation and no matched-budget AutoML control. Its four tasks are not chosen by a stated rule. Its shuffle comparison is part of one task's metric, whereas our C3 is a scrambled-signal condition run on every task.

**BrainBench** (EEG; arXiv 2608.04156)
- *Measures:* LLM systems turning a natural-language instruction plus EEG recordings into a scientific report and artifacts. It has four subsets (foundational analysis, sleep, neurocognitive, physiological integration) drawn from 17 datasets. Validators check numerical, categorical, set, sequence, semantic and artifact correctness. Models run under CodeAct and under BrainAgent.
- *Difference:* EEG only, scored on correctness of analysis outputs, with no reference point from trained task-specific models. We score predictions on held-out test cases against a SOTA rerun and cover several modalities.

**BrainAgent** (arXiv 2606.25400)
- *Measures:* a multi-agent framework, not mainly a benchmark. A supervisor agent dispatches sub-agents with general and EEG-specific tools to turn natural-language requests into EEG analysis pipelines. Its own evaluation set is reported elsewhere as 60 instructions on sleep and emotion.
- *Difference:* it is a system to be evaluated, not a benchmark. We evaluate general agent CLIs without domain scaffolding. A domain framework like BrainAgent could be entered as an evaluated agent under the same harness.

**NeuroAgent** (arXiv 2605.06584)
- *Measures:* a hierarchical multi-agent framework for neuroimaging preprocessing and analysis (sMRI, fMRI, dMRI, PET) with a generate-execute-validate loop and human in the loop. It is evaluated on ADNI for intent-parsing accuracy, preprocessing step correctness and Alzheimer's classification AUC, against single-modality baselines.
- *Difference:* it is a system on one cohort, measuring pipeline correctness. We compare general agents against trained models across tasks and modalities, with controls. The name collides with this project's; see D-002.

## Agent and LLM benchmarks outside neuroscience

**BioXArena** (arXiv 2605.15766)
- *Measures:* 76 end-to-end biomedical ML tasks in 9 domains, most of them multi-modal. Agents write code, train models and predict private test samples against hidden labels. Scores are on a unified 0–100 scale. It compares 18 harness–backbone configurations under a controlled resource setup, with ablations over backbone, scaffold, budget and domain.
- *Difference:* closest in format: build-a-model tasks, hidden labels, controlled resources. But it covers molecular, cellular and clinical biology, not neural data. Its abstract does not describe SOTA reruns per task, a trained-model-tools ablation, or scrambled-signal controls.

**HeaRTS** (arXiv 2603.06638)
- *Measures:* LLM reasoning over health time series: 16 datasets, 12 health domains, 20 signal modalities and 110 tasks across perception, inference, generation and deduction. It reports that LLMs trail specialized models.
- *Difference:* it tests the LLM's reasoning over signals presented to it, with agents as a future direction. We test agents that write and run analysis code, scored against a SOTA rerun on the same split, with the trained models offered as tools.

**MLE-bench** (arXiv 2410.07095)
- *Measures:* ML engineering on 75 Kaggle competitions, graded by Kaggle medal thresholds from public human leaderboards.
- *Difference:* general-domain, with human leaderboards as the reference. We use the field's published SOTA method rerun on our split, neural data only, plus the tools ablation and scrambled controls. Contamination is handled per task through release and cutoff dates and private splits.

**AIRS-Bench** (arXiv 2602.06855)
- *Measures:* 20 tasks from recent ML papers (language modelling, mathematics, bioinformatics, time series). Agents get no baseline code and must train and validate a model. It reports normalized score against the source paper's SOTA, valid-submission rate and Elo, with the AIRA-dojo and MLGym harnesses.
- *Difference:* the closest in normalizing against SOTA. We rerun SOTA on the exact split instead of relying on the reported number, add the C1 − C2 and C3 conditions and the matched-budget AutoML baseline, and restrict to neuroscience modalities chosen by rule.

## Brain-foundation-model benchmarks

**AdaBrain-Bench** (arXiv 2507.09882) and **EEG-FM-Bench** (arXiv 2508.17742)
- *Measure:* standardized evaluation of pretrained EEG foundation models against supervised baselines. AdaBrain-Bench covers 13 datasets and 7 BCI applications in cross-subject, multi-subject and few-shot settings. EEG-FM-Bench covers 14 datasets and 10 paradigms, with several fine-tuning strategies and diagnostic analyses.
- *Difference:* they evaluate trained models, not agents, and only on EEG. To us these models are candidate SOTA baselines and trained-model tools, and their protocols are a source for splits and SOTA numbers.
