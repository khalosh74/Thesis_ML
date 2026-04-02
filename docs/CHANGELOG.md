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

- 2026-04-02 (follow-up):
  - Created `docs/proposals/` and `docs/archived/` directories and scanned `docs/` for remaining developer/design plan files.
  - Result: No additional standalone implementation/design plan files were found in `docs/` to move; canonical governance, operator, and maintainer docs remain in place.
  - Updated todo list to mark scanning and directory creation completed.

-- Recorded by repository agent (follow-up)

## 2026-04-02 — Merged script inventory

- Merged `docs/maintainer/script_inventory.md` into `docs/MAINTAINER_GUIDE.md`.
- Backups created:
  - `docs/archived/script_inventory_backup_2026-04-02.md`
  - `docs/archived/MAINTAINER_GUIDE_backup_2026-04-02.md`
- Replaced `docs/maintainer/script_inventory.md` with a pointer to the merged section and added a merge note in `docs/MAINTAINER_GUIDE.md`.
- Reason: Consolidate maintainer-facing operator/script references into a single canonical maintainer guide.

-- Recorded by repository agent
