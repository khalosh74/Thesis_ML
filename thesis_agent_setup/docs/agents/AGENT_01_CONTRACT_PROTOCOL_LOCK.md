# AGENT 01 — CONTRACT / PROTOCOL LOCK

## Branch
`agent/contract/protocol-lock`

## Mission
Make the thesis scientific contract explicit, complete, and unambiguous.

This agent exists to ensure that the repository’s official scientific behavior is defined in frozen manifests/contracts rather than being partially hidden in adapters, defaults, or runtime interpretation.

## Scientific Role
This agent protects the integrity of the thesis estimand.

It is responsible for making sure that:
- the primary confirmatory protocol is explicit
- evidence families are explicit
- claim boundaries are encoded
- official science is contract-driven
- no adapter-only hidden defaults remain in official confirmatory paths

## In Scope
- thesis program manifests
- release bundle manifests
- science-contract schemas
- evidence-family declarations
- evidence-level labeling fields
- claim-boundary fields and metadata
- protocol identifiers and versioning
- explicit declaration of primary vs secondary vs robustness vs diagnostic analyses
- formal deviation logging or contract amendment hooks
- documentation for how the frozen contract works

## Out of Scope
- implementing new analyses
- improving model performance
- writing negative controls
- writing robustness analyses
- writing transfer logic
- interpretability outputs
- synthesis outputs beyond what is required to define contracts

## Owned Areas
Suggested ownership:
- `releases/`
- program-level manifest locations
- release schemas / manifest schemas
- contract validation logic
- documentation related to official scientific contract structure

## Read-Only Dependencies
- execution/runtime code
- data contracts
- report generation
- test suite outside owned scope

## Required Outputs
1. Explicit top-level thesis program contract.
2. Explicit evidence-family metadata.
3. Explicit evidence-level tags.
4. Explicit confirmatory protocol record.
5. Explicit prohibition of hidden scientific defaults in official paths.
6. Documentation describing how official science is frozen and audited.

## Required Tests
- schema/manifest validation tests
- tests that confirm official bundles cannot omit required scientific fields
- tests that prevent confirmatory bundles from inheriting hidden defaults
- tests that evidence-level labels are required and valid

## Merge Criteria
- all official science used by the thesis is explicit in the contract
- no hidden confirmatory assumptions remain in adapters
- all new fields are validated
- no runtime behavior changes primary science without manifest support

## Review Checklist
- Did this agent make the science more explicit?
- Did it avoid changing the science itself unless explicitly required?
- Did it reduce ambiguity about what is primary, secondary, robustness, or diagnostic?

## Notes for the Agent
Favor explicitness over convenience.
A small increase in verbosity is acceptable if it makes the confirmatory path auditable.
Do not make open-ended “cleanup” changes outside contract scope.
