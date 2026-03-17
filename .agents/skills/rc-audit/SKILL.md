---
name: rc-audit
description: Run the full release-candidate audit for this repository. Use when the user asks to verify release readiness, audit the repo after RC-1, confirm everything is working properly, or assess whether the project is ready for the frozen experiment campaign.
---

## When to use this

Use this skill when the user wants a full verification pass of the repository as a release candidate.

Use it for:
- release-candidate readiness checks
- full repo audits
- “is everything working properly?”
- post-RC-1 verification
- pre-experiment freeze validation

Do not use it for implementing new features.

## What to do

1. Read the repo-level `AGENTS.md` first and follow all official workflow rules.
2. Inspect the main docs and workflow definitions:
   - `README.md`
   - `docs/ARCHITECTURE.md`
   - `docs/RUNBOOK.md`
   - `docs/OPERATOR_GUIDE.md`
   - `docs/RELEASE.md`
3. Run the main validation commands if available:
   - tests
   - release hygiene check
   - performance smoke
   - artifact verification
   - one comparison validation path
   - one confirmatory protocol validation path
4. Inspect emitted artifacts directly. Do not rely only on exit codes.
5. Report using these labels:
   - verified
   - partially verified
   - not verified
   - failed

## Required checks

Check all of these if possible:
- repository/release hygiene
- contract strictness
- deterministic reproducibility
- artifact completeness
- failure behavior and observability
- performance sanity
- architecture/modularity sanity
- governance/mode separation

## Report format

Return:
1. Executive verdict
2. Verification matrix
3. Commands executed
4. Artifacts inspected
5. Failures and weaknesses
6. Release blockers
7. Non-blocking improvements
8. Final recommendation

## Important rules

- Verify, do not improve, unless the user explicitly asks.
- Distinguish clearly between what was actually confirmed and what was only inferred.
- If the environment blocks a check, say exactly what prevented confirmation.