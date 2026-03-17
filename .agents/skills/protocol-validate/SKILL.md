---
name: protocol-validate
description: Validate the canonical protocol workflow in this repository. Use when the user wants to test or verify `thesisml-run-protocol`, run the canonical protocol in dry-run mode, run a small real official suite, or inspect confirmatory protocol artifacts.
---

## When to use this

Use when the task is specifically about validating the confirmatory protocol path.

Examples:
- “validate the canonical protocol”
- “run the protocol dry-run”
- “check confirmatory artifacts”
- “verify the official thesis protocol path”

## What to do

1. Read `AGENTS.md` and respect confirmatory-mode rules.
2. Identify the relevant protocol file under `configs/protocols/`.
3. Prefer this sequence:
   - dry-run the protocol first
   - then run one small real suite if requested or appropriate
4. Inspect protocol-level artifacts and one run-level artifact directory.
5. Confirm:
   - framework mode is confirmatory
   - canonical/protocol metadata is present
   - required artifacts exist
   - metric/methodology policy is visible
   - run status is machine-readable

## Typical commands

Use the official command shape:
- `thesisml-run-protocol --protocol <protocol.json> --all-suites --dry-run`
- `thesisml-run-protocol --protocol <protocol.json> --suite <suite_name>`

## What to inspect

At minimum inspect:
- protocol source copy
- compiled protocol manifest
- suite summary
- execution status
- report index
- one run-level `config.json`
- one run-level `metrics.json`

## Important rules

- Do not replace protocol execution with exploratory commands.
- Do not pass science-affecting overrides unless the repository explicitly supports them in confirmatory mode.
- If a required artifact is missing, report it as a validation issue.