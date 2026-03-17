---
name: deterministic-rerun
description: Verify deterministic reproducibility by rerunning the same official command twice and comparing the meaningful outputs. Use when the user wants to confirm reproducibility, compare reruns, or validate that the same protocol/config/seed/code revision produces the same official results.
---

## When to use this

Use when the task is about reproducibility, rerun stability, or deterministic validation.

Examples:
- “check deterministic reproducibility”
- “rerun this protocol twice and compare”
- “validate that official runs are reproducible”

## What to do

1. Read `AGENTS.md` and prefer official comparison or confirmatory paths.
2. Choose one small official path if the user did not specify one.
3. Run the exact same command twice with:
   - same code revision
   - same protocol/comparison config
   - same seed
   - same dataset/index/cache inputs
4. Compare meaningful outputs after ignoring expected non-deterministic fields like timestamps if necessary.

## What to compare

Prefer comparing:
- metrics
- predictions
- split manifests
- comparison decision
- effective metric/methodology policy
- key artifact hashes or normalized contents

## Report format

Return:
- command executed
- run paths compared
- outputs compared
- exact match / normalized match / mismatch
- suspected cause if mismatch exists

## Important rules

- Do not claim determinism unless outputs were actually compared.
- Separate expected timestamp/path differences from real scientific output drift.
- Use official comparison or confirmatory paths, not exploratory runs, unless the user explicitly asks otherwise.