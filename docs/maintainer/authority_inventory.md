# Authority Inventory

This inventory classifies truth-bearing and related files by authority layer.

## Scientific
- `configs/confirmatory/confirmatory_scope_v1.json`
- `docs/Science/CLAIM_EVALUATION_RULES.md`
- `docs/Science/THESIS_TRACEABILITY_MATRIX.md`

## Runtime
- `configs/decision_support_registry_revised_execution.json` (thesis runtime authority)
- `configs/decision_support_registry.json` (package/demo runtime default)
- `configs/config_registry.json` (alias and lifecycle governance for runtime resolution)

## Generation
- `templates/thesis_experiment_program.xlsx` (governed template)
- `workbooks/thesis_program_instances/thesis_experiment_program_revised_v1.xlsx` (study workbook instance)
- `src/Thesis_ML/workbook/template_constants.py`
- `src/Thesis_ML/workbook/template_builder.py`
- `scripts/freeze_workbook_registry.py`

## Derived
- `src/Thesis_ML/assets/configs/decision_support_registry.json` (packaged mirror)
- `src/Thesis_ML/assets/templates/thesis_experiment_program.xlsx` (packaged mirror)

## Archive
- `configs/archive/registries/` (backup registries root)
- `configs/archive/registries/decision_support_registry_revised_execution.E02_backup.json` (legacy backup file)

## Machine-Readable Layer Contract
- `configs/authority_manifest.json` is the machine-readable declaration of the scientific/runtime/generation/derived/archive authority model.
