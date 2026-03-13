# Acceptance Suite

This directory contains a tiny deterministic golden path for the framework:

1. Generate and populate a workbook fixture.
2. Compile workbook rows into an internal manifest.
3. Execute a real tiny run on synthetic local fMRI-like fixtures.
4. Write results back to a versioned workbook copy.
5. Assert key machine-written outputs in `Trial_Results`, `Summary_Outputs`, and `Run_Log`.

Fixtures are intentionally small so this suite is CI-friendly and does not require external systems.
