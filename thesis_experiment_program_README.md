# Thesis Experiment Program Workbook (v2)

## Purpose
`thesis_experiment_program.xlsx` is the thesis experiment governance workbook for:
- pre-interpretation experiment specification,
- explicit separation of confirmatory, decision-support, and exploratory evidence,
- lock-traceable method decisions,
- leakage-aware split and interpretation discipline,
- direct mapping from experiment outputs to thesis writing sections.

It is intended as the single source of truth for method choice, method application, robustness support, discussion/limitations traceability, and governance logs.

## Sheet structure (11 sheets)
1. **README**  
   Rules, evidence tiers, freeze policy, stage flow, and workbook workflow.
2. **Master_Experiments**  
   Main control sheet with E01-E19, strict readiness logic, and experiment governance fields.
3. **Run_Log**  
   Run-level execution registry with reproducibility metadata and drift-control flags.
4. **Decision_Log**  
   Decision lock registry for target/split/model/feature-preprocessing and exploratory scope.
5. **Confirmatory_Set**  
   Confirmatory/supporting subset with eligibility, run-readiness, completion, and Chapter 4 readiness tracking.
6. **Thesis_Map**  
   Mapping from experiments to thesis chapters/sections and claim support role.
7. **Dictionary_Validation**  
   Controlled vocabularies for dropdowns plus concise methodological definitions.
8. **Dashboard**  
   Formula-driven overview of experiment, decision, claim, AI, ethics, and confirmatory readiness status.
9. **Claim_Ledger**  
   Claim discipline sheet linking claims to supporting experiments and evidence status.
10. **AI_Usage_Log**  
    AI/tool-use transparency log with human verification status and thesis-use flag.
11. **Ethics_Governance_Notes**  
    Ethics/governance risk, mitigation, section linkage, and status tracking.

## Stage system (v2)
- Stage 1 - Target lock  
- Stage 2 - Split lock  
- Stage 3 - Model lock  
- Stage 4 - Feature/preprocessing lock  
- Stage 5 - Confirmatory analysis  
- Stage 6 - Robustness analysis  
- Stage 7 - Exploratory extension

This stage flow is used consistently in controlled vocabularies, experiment rows, and decision tracking.

## Confirmatory gating model
The workbook separates two checkpoints:
- **Confirmatory eligibility achieved?**  
  Requires decision locks (D01-D07 in `Decision_Log`).
- **Confirmatory ready for Chapter 4?**  
  Requires eligibility plus required confirmatory/supporting entries completed and explicitly marked `Ready_for_Chapter_4 = YES`.

This distinction prevents premature reporting when method locks exist but reporting prerequisites are incomplete.

## Run and governance usage
1. Define/maintain experiment specification in **Master_Experiments** before interpreting results.
2. Register each execution in **Run_Log** (with commit/config/artifact path).
3. Record method locks in **Decision_Log** only when evidence criteria are satisfied.
4. Track confirmatory gating in **Confirmatory_Set**.
5. Keep claims explicit and bounded in **Claim_Ledger**.
6. Record AI assistance and human verification in **AI_Usage_Log**.
7. Track ethics/governance concerns in **Ethics_Governance_Notes**.
8. Use **Dashboard** for readiness checks before Chapter 4 reporting.

## Assumptions and non-invention policy
- No numerical results, locked decisions, or completion statuses are fabricated.
- Defaults remain conservative (`Planned`, `Open`, blank outcomes) until evidence is entered.
- Placeholders are used where grounding links are not yet finalized (e.g., source packet linkage).
- Workbook regeneration is deterministic via `create_thesis_experiment_workbook.py`.
