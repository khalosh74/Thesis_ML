# Branch Setup Status

- Base branch used: `master`
- Branches created:
  - `program/thesis-evidence`
  - `agent/contract/protocol-lock`
  - `agent/confirmatory/main-claim`
  - `agent/falsification/negative-controls`
  - `agent/robustness/split-strength`
  - `agent/transfer/cross-person`
  - `agent/interpretability/stability`
  - `agent/synthesis/evidence-pack`
  - `agent/quality/invariants`
- Branches not created: none
- Manual protections or GitHub actions still needed:
  - protect `main` and `program/thesis-evidence` on GitHub
  - enforce PR-only merges for the program and agent branches
  - add required review / status checks before merge if not already configured
  - create separate worktrees or checkouts for each agent branch before implementation work
