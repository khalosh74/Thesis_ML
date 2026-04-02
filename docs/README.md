# Documentation Index

Core docs:

- `ARCHITECTURE.md`: package/module boundaries and runtime flows.
- `OPERATOR_GUIDE.md`: concise canonical operator path + gold acceptance command.
- `MAINTAINER_GUIDE.md`: concise maintainer surface and closure policy.
- `RUNBOOK.md`: operator commands for install, run, rerun/resume, and health checks.
 - `ARCHITECTURE.md`: package/module boundaries and runtime flows.
- `RUNBOOK.md`: operator commands for install, run, rerun/resume, and health checks. (Now includes the canonical Operator Guide content.)
- `MAINTAINER_GUIDE.md`: concise maintainer surface, closure policy, and script inventory (script inventory merged here).
- `WORKBOOK_WORKFLOW.md`: workbook generation, compile, execution, and write-back.
- `SEGMENT_EXECUTION.md`: start/end section execution model and base artifact rules.
- `EXTENDING.md`: how to add a section or workbook sheet safely.
- `SCHEMA_MIGRATIONS.md`: schema versioning and migration policy.

Existing focused notes:

- `EXPERIMENTS.md`: experiment-mode semantics and leakage constraints.
- `DECISION_SUPPORT_AUTOMATION.md`: campaign orchestration behavior and outputs.
- `RELEASE.md`: release gate process.
- `PRIVACY_AND_DATA_HANDLING.md`: privacy handling and publication/redaction expectations.
- `USE_AND_MISUSE_BOUNDARIES.md`: intended use, claim boundaries, and misuse warnings.
- `CONFIRMATORY_READY.md`: explicit governance criteria for confirmatory-ready status.

Consolidation notes (2026-04-02):

- `OPERATOR_GUIDE.md` content was merged into `RUNBOOK.md`. A backup of the original standalone operator guide is at `docs/archived/OPERATOR_GUIDE_backup_2026-04-02.md`.
- The maintainer-facing `script_inventory.md` was merged into `MAINTAINER_GUIDE.md`. Backup: `docs/archived/script_inventory_backup_2026-04-02.md`.
- Developer/design proposals and plan files were moved out of the top-level index or archived; `docs/proposals/` and `docs/archived/` were created to hold such materials.

See `docs/CHANGELOG.md` for a recorded list of deleted/merged files and rationale.
