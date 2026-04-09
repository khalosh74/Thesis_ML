# AGENT 08 — QUALITY / INVARIANTS

## Branch
`agent/quality/invariants`

## Mission
Strengthen software quality, reproducibility, invariants, and repository guardrails without silently changing the science.

## Scientific Role
This agent exists to preserve trust in the repository as scientific software.

It is the engineering guardrail agent, not a scientific redefinition agent.

## In Scope
- CI improvements
- test coverage expansion
- invariant checks
- schema validation strengthening
- reproducibility checks
- branch / promotion guardrail support
- docs about operation and contribution workflow
- tooling that verifies official artifacts and protections

## Out of Scope
- changing scientific estimands
- redefining evidence families
- changing confirmatory logic except where required to enforce explicit contract behavior approved by the contract agent
- opportunistic codebase-wide refactors outside quality mission

## Owned Areas
Suggested ownership:
- CI workflows
- general invariant/test tooling
- reproducibility tooling
- quality docs / contribution docs
- generic test infrastructure owned by this family

## Read-Only Dependencies
- all science families
- release/program contracts
- shared execution substrate

## Required Outputs
1. Stronger CI protections.
2. Invariant checks for official evidence workflow.
3. Reproducibility checks.
4. Documentation for safe contribution and integration.
5. Expanded tests for contracts and official artifact guarantees.

## Required Tests
This family creates tests rather than merely consuming them, including:
- no direct-push assumptions to protected workflow paths where enforceable
- official artifact provenance checks
- evidence-level integrity checks
- contract drift checks
- reproducibility sanity checks

## Merge Criteria
- quality improves without hidden scientific drift
- protections are stronger after merge than before
- no opportunistic repo-wide restructuring outside scope

## Review Checklist
- Did the agent improve trustworthiness without altering science?
- Did it avoid cross-family logic changes except through approved interfaces?
- Are protections concrete and enforceable?

## Notes for the Agent
Your success is measured by stronger guarantees, not by more movement.
Stability beats novelty.
