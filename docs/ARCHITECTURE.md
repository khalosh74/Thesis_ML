# Architecture

## Public CLIs

- `thesisml-validate-dataset`
- `thesisml-validate-release`
- `thesisml-run-release`
- `thesisml-promote-run`

## Core Modules

- `src/Thesis_ML/release/models.py`: typed release, dataset, run, and promotion contracts
- `src/Thesis_ML/release/loader.py`: bundle and manifest loading
- `src/Thesis_ML/release/validator.py`: release and dataset validation
- `src/Thesis_ML/release/scope.py`: strict scope compilation and scope-alignment verification
- `src/Thesis_ML/release/runner.py`: candidate/exploratory/scratch execution orchestration
- `src/Thesis_ML/release/evidence.py`: evidence verification
- `src/Thesis_ML/release/promotion.py`: candidate to official promotion

## Runtime Shape

`science.json` defines the scope.

The runner compiles exact selected rows to `selected_samples.csv` and executes on that subset.

The run fails if runtime-selected rows diverge from the compiled sample-id set.

## Evidence Contract

Candidate promotability requires:

- passed scope alignment verification
- passed evidence verification
- required run/release artifacts present

Official runs are immutable by policy and singleton per release id.
