# Privacy and Data Handling

This repository is a scientific ML framework, not a public data host. Governance for
privacy/data handling is mandatory for official runs.

## 1) Data classes

- Internal sensitive data:
  - subject-level fMRI inputs and linked identifiers.
  - any dataset index that can re-identify subjects directly or indirectly.
- Synthetic or test fixture data:
  - generated test arrays/indexes with no real participant linkage.
- Derived run artifacts:
  - metrics, manifests, reports, and summaries produced by runs.
- Shareable governance metadata:
  - schema files, protocol/comparison configs, docs, and validation summaries that
    contain no sensitive subject-level material.

## 2) What must not be published in release bundles

- Raw neuroimaging data and subject-linked source files.
- Dataset roots/cache roots copied from local machines.
- Dataset index files containing unapproved participant identifiers.
- Any artifact that includes direct identifiers or unredacted file paths into local
  restricted storage.

## 3) Handling expectations for paths and identifiers

- Treat `index_csv`, `data_root`, and `cache_dir` as sensitive operational metadata.
- Prefer redacted/relative path presentation in release-facing materials.
- Do not introduce new public artifacts that expose raw local absolute paths unless
  there is a documented scientific or audit requirement.
- Subject/session/task identifiers must be handled under the project's approved data
  governance process.

## 4) Artifact redaction and review

Before publication or external sharing:

1. Run release hygiene and official artifact verification.
2. Review run/config/report artifacts for path and identifier leakage.
3. Remove or redact non-essential sensitive metadata.
4. Keep confirmatory provenance fields needed for reproducibility audit, but do not
   publish restricted source data.

## 5) External dataset handling

- External validation is compatibility-only in this phase.
- External compatibility outputs are governance artifacts, not external performance
  evidence.
- Any future external performance claims require explicit protocol/policy updates and
  separate governance approval.

## 6) Relationship to confirmatory outputs

Confirmatory-ready status does not override privacy/data-handling obligations.
Governance checks are additive to scientific contract checks.
