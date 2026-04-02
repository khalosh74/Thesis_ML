# Changelog

## 2026-04-02 — Documentation cleanup

- Deleted files (per user request):
  - `docs/GPU_ACCELERATION_PLAN.md` — GPU backend implementation plan (developer/design doc).
  - `docs/confirmatory/confirmatory_plan_v1.md` — locked confirmatory plan (removed as requested).

- Reason: User requested removal of implementation/design plan documents from the public `docs/` tree.

- Actions taken:
  1. Removed the two files from `docs/` on 2026-04-02.
  2. Created this changelog entry to record the deletion.
  3. Recommended next steps (not applied here):
     - Create `docs/proposals/` and `docs/archived/` for developer/design materials.
     - Move remaining implementation/design docs into `docs/proposals/`.
     - Merge `OPERATOR_GUIDE.md` into `RUNBOOK.md` (create backup first).
     - Merge `docs/maintainer/script_inventory.md` into `MAINTAINER_GUIDE.md`.
     - Update `docs/README.md` to reflect consolidated structure.
     - Run `python scripts/release_hygiene_check.py` to verify doc requirements.

- Recovery note: Deleted files remain recoverable from Git history (use revert/checkout).

-- Recorded by repository agent
