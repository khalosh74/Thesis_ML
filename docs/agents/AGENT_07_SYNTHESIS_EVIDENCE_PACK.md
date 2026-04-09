# AGENT 07 — SYNTHESIS / EVIDENCE PACK

## Branch
`agent/synthesis/evidence-pack`

## Mission
Build the thesis-ready synthesis layer that converts promoted official outputs into structured evidence artifacts for writing, review, and final reporting.

## Scientific Role
This agent protects the alignment between code and thesis narrative.

It ensures that the thesis uses only official evidence and that all claims are traceable to promoted outputs.

## In Scope
- claim ledger generation
- decision map generation
- result-to-RQ mapping
- figure pack generation from official outputs
- table pack generation from official outputs
- limitations register generation
- thesis evidence pack assembly
- provenance and audit summaries tied to official artifacts

## Out of Scope
- generating new science through hidden post-processing
- redefining metrics or claims
- editing scientific contracts
- engineering-only repo hygiene work

## Owned Areas
Suggested ownership:
- synthesis/report assembly modules
- evidence pack outputs
- thesis export/report generation owned by this family
- synthesis family tests

## Read-Only Dependencies
- all promoted evidence families
- contract definitions
- claim boundaries

## Required Outputs
1. Claim ledger.
2. Decision map.
3. Results-to-RQ map.
4. Figure pack.
5. Table pack.
6. Limitations register.
7. Final thesis evidence bundle / pack.

## Required Tests
- tests that only official promoted outputs are accepted as synthesis inputs
- tests that evidence-level metadata is preserved in synthesis
- tests that secondary/diagnostic artifacts cannot silently enter confirmatory summaries

## Merge Criteria
- synthesis consumes only official promoted evidence
- outputs are traceable and auditable
- no hidden transformations create new science beyond source outputs

## Review Checklist
- Could every thesis-facing artifact be traced back to an official promoted source?
- Does the synthesis preserve claim boundaries?
- Does the synthesis reduce writing ambiguity?

## Notes for the Agent
You are not writing the thesis text itself.
You are building the evidence substrate that the thesis can trust.
