# AGENT 04 — ROBUSTNESS / SPLIT STRENGTH

## Branch
`agent/robustness/split-strength`

## Mission
Build the robustness family that tests whether the main conclusion survives stronger notions of “unseen” and restricted scientific scopes.

## Scientific Role
This agent answers: if the held-out condition is made more demanding, does the conclusion hold, weaken, or fail?

## In Scope
- leave-one-day-out analyses
- chronological early-train/late-test analyses
- task-restricted analyses
- modality-restricted analyses
- sensitivity to session grouping / spacing / stimulus reuse if explicitly approved
- bounded robustness reports
- family-specific validation and tests

## Out of Scope
- redefining the primary claim
- open-ended multiverse search
- exploratory performance maximization
- transfer work
- falsification controls unless explicitly shared through a declared interface

## Owned Areas
Suggested ownership:
- robustness bundle/config definitions
- robustness family reports
- robustness family tests

## Read-Only Dependencies
- confirmatory family semantics
- contract definitions
- shared execution substrate

## Required Outputs
1. Official robustness bundles.
2. Robustness reports comparing stronger holdouts to the primary confirmatory setting.
3. Clear labels indicating which robustness checks are supporting evidence and which are merely delimitations.

## Required Tests
- tests that stronger split/grouping logic is enforced correctly
- tests that robustness outputs are separated from confirmatory outputs
- tests that restricted-scope analyses preserve valid dataset structure

## Merge Criteria
- stronger holdouts are implemented exactly as declared
- outputs describe support, weakening, or delimitation clearly
- no accidental scope expansion occurs

## Review Checklist
- Are these checks genuinely stronger or meaningfully different from the confirmatory split?
- Is the interpretation of robustness evidence bounded and disciplined?

## Notes for the Agent
Robustness is not a search for a better number.
It is a disciplined test of whether the original conclusion survives under harder conditions.
