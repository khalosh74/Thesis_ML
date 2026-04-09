# AGENT 03 — FALSIFICATION / NEGATIVE CONTROLS

## Branch
`agent/falsification/negative-controls`

## Mission
Build the adversarial evidence family that attempts to invalidate or weaken the main claim through explicit controls.

## Scientific Role
This agent serves as the skeptical reviewer inside the repository.

It exists to answer: could the apparent main effect arise from leakage, nuisance structure, grouping artifacts, chronology, or structurally irrelevant signal?

## In Scope
- nuisance-only model family
- grouped shuffled-label controls
- session/day/order prediction controls where scientifically approved
- weak-split demonstration analyses if predeclared
- spatial/feature negative controls where appropriate
- explicit control reports summarizing whether the main interpretation survives
- family-specific tests for control correctness

## Out of Scope
- improving the main model
- changing the confirmatory path
- redefining the thesis estimand
- open-ended exploratory debugging outside the control family

## Owned Areas
Suggested ownership:
- falsification bundle/config definitions
- control analysis modules owned by this family
- control reports and summaries
- control family tests

## Read-Only Dependencies
- confirmatory contract
- shared data and execution substrate
- promoted confirmatory semantics

## Required Outputs
1. Official falsification bundles.
2. Group-aware shuffled-label control outputs.
3. Nuisance/control model outputs.
4. A structured falsification summary describing which alternative explanations were challenged and with what result.

## Required Tests
- tests that controls preserve required grouping structure
- tests that shuffled-label controls do not break the repeated-measures assumptions incorrectly
- tests that nuisance-only analyses use only permitted nuisance inputs
- tests that control artifacts are labeled correctly and cannot be mistaken for confirmatory outputs

## Merge Criteria
- controls are scientifically meaningful and grouped correctly
- no confirmatory artifacts are overwritten or reinterpreted
- outputs clearly indicate challenge-vs-support status

## Review Checklist
- Does this family genuinely try to break the main interpretation?
- Are controls properly separated from the confirmatory path?
- Are the control semantics precise enough for thesis discussion?

## Notes for the Agent
Think adversarially, not sympathetically.
Your success is measured by how seriously you test the claim, not by whether the main result survives.
