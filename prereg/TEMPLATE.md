# Pre-registration: <<FILL: batch name>>

- Authors: <<FILL>>
- Date: <<FILL: YYYY-MM-DD>>
- Tasks and their frozen spec commits: <<FILL>>
- Models, CLI versions and knowledge cutoffs: <<FILL>>

## Hypotheses

<<FILL: what each condition contrast (C1 vs C2, C3 vs baseline, C4, C5) is expected to show, and why>>

## Analysis

- Primary outcome: normalized score vs SOTA per task and condition, paired bootstrap over test cases (`docs/DESIGN.md`).
- Secondary: normalized score vs noise ceiling; C1 − C2 and C1 − C3 gaps; costs.
- Leakage check: any C3 lower CI bound above 0 is treated as leakage and the task is investigated before its other results are reported.
- Exclusions: <<FILL: none, or the rule>>
- What counts as complete: every planned run and baseline scored (the scorer's completeness rule).

## Plan and predictions

```yaml
plan:
  seed: <<FILL: integer>>
  blocks:
    - tasks: [<<FILL>>]
      conditions: [C1, C2, C3]
      models: [<<FILL>>]
      reps: <<FILL: at most the task's submission_limit>>
predictions:
  <<FILL: named thresholds the report is checked against>>
```

## Deviations

Any deviation after generation is recorded in `docs/DECISIONS.md` with a date and reason; this file is not edited after the batch exists.
