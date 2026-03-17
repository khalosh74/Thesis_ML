---
name: artifact-verify
description: Verify official run artifacts for completeness and consistency. Use when the user wants to inspect output directories, check whether required artifacts exist, validate protocol or comparison outputs, or confirm that official artifacts match the repository’s invariant artifact contract.
---

## When to use this

Use for artifact-focused validation tasks.

Examples:
- “check the artifacts”
- “verify this run output”
- “confirm the output folder is complete”
- “validate the official artifacts”

## What to do

1. Read `AGENTS.md` to understand official artifact expectations.
2. Identify the mode:
   - exploratory
   - locked comparison
   - confirmatory
3. Verify the artifact set appropriate to that mode.
4. Prefer repository scripts/utilities for artifact verification if they exist.
5. If no validator exists for a specific artifact, inspect the files directly.

## What to verify

Where applicable, check for:
- config/provenance artifact
- metrics artifact
- fold metrics
- fold splits
- predictions
- subgroup outputs
- tuning outputs
- interpretability outputs
- run/execution status
- comparison or protocol manifest
- comparison decision or suite summary
- report index

## Report format

Return:
- path inspected
- mode detected
- artifacts found
- artifacts missing
- suspicious or inconsistent contents
- verdict: valid / incomplete / inconsistent / failed

## Important rules

- Do not assume success from a directory existing.
- Inspect actual contents.
- Treat partial official artifact sets as a real issue.