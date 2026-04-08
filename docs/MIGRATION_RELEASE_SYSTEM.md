# Migration: Release-System Hard Cutover

Status: complete.

Removed authority files:

- `configs/authority_manifest.json`
- `configs/config_registry.json`
- `configs/decision_support_registry*.json`
- `configs/protocols/*.json`
- `configs/comparisons/*`
- `configs/archive/*`

Removed legacy runtime surfaces:

- old runner CLIs and modules for comparison/workbook/decision-support
- old protocol/comparison/orchestration/workbook package trees

Active authority precedence:

`release.json > science.json > dataset manifest > environment.json > execution.json > evidence.json`

Active thesis-final command surface:

- `thesisml-validate-dataset`
- `thesisml-validate-release`
- `thesisml-run-release`
- `thesisml-promote-run`
