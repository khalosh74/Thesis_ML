# AGENTS.md

## Repository Mode

This repository is a release-only scientific execution system.

Active thesis-final command surface:

- `thesisml-validate-dataset`
- `thesisml-validate-release`
- `thesisml-run-release`
- `thesisml-promote-run`

## Non-negotiable

1. `releases/thesis_final_v1/science.json` is the only scientific scope authority.
2. Scope must be compiled to `selected_samples.csv` before compute.
3. Runtime execution must use the compiled sample subset exactly.
4. Candidate runs become official only through promotion.
5. Exactly one official run is allowed per release id.
6. CPU-only policy is mandatory for official release execution.

## Working Rules

- Preserve model and evaluation math.
- Prefer additive, typed, explicit changes.
- Keep evidence outputs machine-readable.
- Do not reintroduce parallel runtime authority trees.
- Do not add legacy compatibility wrappers.
