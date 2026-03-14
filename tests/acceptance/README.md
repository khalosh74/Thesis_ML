# Acceptance Suite

This directory contains a tiny deterministic golden path for the framework:

1. Generate and populate a workbook fixture.
2. Compile workbook rows into an internal manifest.
3. Execute a real tiny run on synthetic local fMRI-like fixtures.
4. Write results back to a versioned workbook copy.
5. Assert key machine-written outputs in `Trial_Results`, `Summary_Outputs`,
   factorial sheets (`Generated_Design_Matrix`, `Effect_Summaries`), and `Run_Log`.

Fixtures are intentionally small so this suite is CI-friendly and does not require external systems.

Canonical designed-study path:

- uses `templates/examples/canonical_designed_study.xlsx`
- validates workbook schema support before compilation
- asserts study-review summary generation and guardrail disposition visibility
- asserts generated design matrix, trial write-back, and descriptive effect summaries
