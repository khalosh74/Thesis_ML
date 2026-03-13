# Schema and Version Migration Notes

Schema/version constants are centralized in:

- `src/Thesis_ML/config/schema_versions.py`

Current public contract versions:

- `workbook_schema_version`
- `compiled_manifest_schema_version`
- `artifact_schema_version`
- `workbook_writeback_schema_version`
- `summary_result_schema_version`

## Where versions are enforced

1. Workbook metadata
- README sheet deterministic metadata block
- Writer/reader:
  - `src/Thesis_ML/workbook/schema_metadata.py`

2. Workbook compiler
- Validates supported workbook schema versions
- `src/Thesis_ML/orchestration/workbook_compiler.py`

3. Compiled manifest contracts
- Validates compiled manifest schema version
- `src/Thesis_ML/orchestration/contracts.py`

4. Artifact registry records
- Persist `artifact_schema_version` per artifact row
- `src/Thesis_ML/artifacts/registry.py`

5. Result aggregation / write-back metadata
- Uses summary/write-back schema versions in outputs and workbook metadata

## Migration policy

When changing a public schema:

1. Add a new version constant in `schema_versions.py`.
2. Add it to corresponding `SUPPORTED_*` set.
3. Update serializers/writers to emit the new version.
4. Update readers/validators for compatibility behavior.
5. Add migration tests:
   - supported version succeeds
   - unsupported version fails with clear error
6. Update this document and `README.md` release notes section.

## Backward compatibility guidance

- Prefer additive changes over destructive renames/removals.
- Keep at least one prior version readable during migration windows when feasible.
- Use explicit validation errors; do not silently coerce unknown versions.

