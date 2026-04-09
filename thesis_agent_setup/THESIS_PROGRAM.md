# THESIS_PROGRAM.md

## 1. Purpose

This document is the frozen operating contract for the thesis program.

Its purpose is to ensure that all work in the repository serves the thesis as a **scientific evidence program** rather than as a generic experiment platform. It defines the objective, claim boundaries, evidence hierarchy, branch structure, agent ownership, merge authority, and non-negotiable operating rules.

All agents, humans, Copilot sessions, Codex sessions, and automation working on this repository must follow this document.

This file is the source of truth for:
- what the thesis is trying to prove
- what the codebase is allowed to claim
- how work is split across parallel agents
- who may edit which parts of the repository
- how changes are reviewed and merged

If there is any conflict between an agent’s local plan and this document, this document wins.

---

## 2. Thesis Objective

The project exists to support a master’s thesis in computer and systems sciences on **trustworthy machine learning for repeated-session fMRI decoding**.

The core contribution is **methodological**, not architectural.

The repository is intended to produce a leakage-aware, interpretation-aware, release-governed thesis evidence program that can distinguish:
- stable within-person signal
- cross-person transfer under domain shift
- nuisance/confounded signal
- robustness to stronger holdout definitions
- explanation stability versus decorative interpretation

The repository is **not** primarily intended to:
- maximize classifier performance at any cost
- conduct open-ended benchmark hunting
- support broad population generalization claims from N = 2
- support clinical or causal claims
- support naive localization claims from model coefficients

---

## 3. Frozen Scientific Position

### 3.1 Field positioning

The thesis contributes to **computer and systems sciences**, specifically to:
- trustworthy machine learning
- evaluation methodology for repeated-measures scientific data
- release-governed, auditable scientific software for high-dimensional neuroimaging analysis

### 3.2 Contribution statement

The repository supports a thesis that contributes an empirically evaluated, leakage-aware modelling and validation framework for repeated-session fMRI decoding that:
- separates within-person held-out-session generalization from cross-person transfer
- treats interpretability as supporting robustness evidence
- distinguishes confirmatory evidence from secondary and diagnostic evidence
- encodes scientific claims through explicit, auditable contracts

### 3.3 Conservative generalization rule

The dataset contains deeply sampled repeated-session data from **two individuals**.

Therefore:
- within-person claims are allowed if supported by locked evaluation
- cross-person transfer claims are allowed only as **cross-case** or **domain-shift** evidence
- population-level generalization claims are forbidden
- clinical deployment claims are forbidden
- causal claims are forbidden

---

## 4. Allowed Claims and Forbidden Claims

### 4.1 Allowed claims

The repository may support the following claim types if evidence is produced through official, promoted outputs:

1. **Primary confirmatory within-person claim**  
   Under a locked, leakage-aware protocol, repeated-session fMRI representations support above-baseline within-person decoding across held-out sessions.

2. **Secondary descriptive transfer claim**  
   A frozen mapping trained on one participant may or may not transfer to the other participant under domain shift; this is interpreted as cross-case transfer evidence only.

3. **Falsification/control claim**  
   The main result survives or fails against explicit nuisance, leakage, grouping, and structural control analyses.

4. **Robustness claim**  
   The core conclusion is strengthened, weakened, or delimited when tested under stronger holdout definitions or restricted scopes.

5. **Interpretability/stability claim**  
   Explanatory structure can be described as stable or unstable supporting evidence about model behavior, subject to explicit interpretation limits.

6. **Software-methodology claim**  
   The project demonstrates a release-governed scientific evidence workflow aligned with rigorous software engineering and evaluation principles.

### 4.2 Forbidden claims

The repository must not be used to support the following claims unless a future contract explicitly supersedes this program:
- causal claims about neural mechanisms
- clinical utility or patient-level deployment claims
- biomarker claims
- population-level generalization claims
- broad universality claims about emotion representation
- direct localization claims from raw discriminative weights
- “state-of-the-art” claims based on internal performance only

---

## 5. Evidence Hierarchy

All official outputs must belong to one and only one evidence level.

### Level A — Primary Confirmatory
Purpose: supports the main thesis claim.

Criteria:
- predeclared in the program contract
- locked protocol
- promoted official output only
- strict claim boundaries

### Level B — Supporting Robustness
Purpose: tests whether the primary conclusion survives stronger or restricted conditions.

Criteria:
- predeclared robustness family
- does not redefine the primary claim
- can strengthen or delimit interpretation

### Level C — Secondary Descriptive
Purpose: answers a scientifically related but distinct question.

Criteria:
- explicitly labeled secondary
- cannot replace confirmatory evidence
- descriptive interpretation only

### Level D — Diagnostic / Excluded from Thesis Claims
Purpose: debugging, exploratory checks, failure analysis, or scope diagnostics.

Criteria:
- useful for engineering or understanding
- not allowed as thesis-final claim evidence
- may appear only if clearly labeled diagnostic

No output may silently move between levels.

---

## 6. Evidence Families

The thesis program is divided into the following evidence families.

### 6.1 Confirmatory Family
Scope:
- the main within-person claim only
- locked protocol only
- official promoted result only

Out of scope:
- transfer
- open-ended model search
- unbounded exploratory comparison
- interpretability extras not required for the main report

### 6.2 Falsification Family
Scope:
- nuisance-only models
- shuffled-label grouped controls
- session/day/order controls
- weak-split demonstrations if predeclared
- other explicit attempts to invalidate the main interpretation

Out of scope:
- improving the main score
- replacing the main pipeline

### 6.3 Robustness Family
Scope:
- leave-one-day-out
- chronological holdout
- task-restricted analyses
- modality-restricted analyses
- stimulus-reuse or spacing sensitivity analyses
- bounded multiverse checks if explicitly approved

Out of scope:
- open-ended pipeline search

### 6.4 Transfer Family
Scope:
- frozen cross-person transfer only
- directional analysis A→B and B→A
- transfer asymmetry description

Out of scope:
- promotion to primary evidence
- population generalization claims

### 6.5 Interpretability Family
Scope:
- explanation generation
- explanation stability
- representative pattern views
- interpretation safeguards

Out of scope:
- direct neural localization claims
- unsupported mechanistic claims

### 6.6 Synthesis Family
Scope:
- claim ledger
- decision map
- result-to-RQ mapping
- thesis figure/table packs
- limitations register
- evidence pack generation

Out of scope:
- creating new science through hidden post-processing

### 6.7 Quality Family
Scope:
- tests
- CI
- invariant checks
- schema validation
- reproducibility enforcement
- repo hygiene and docs

Out of scope:
- changing the science implicitly

---

## 7. Branch Architecture

### 7.1 Permanent branches
- `main` — protected production branch; no agent may push directly
- `program/thesis-evidence` — protected integration branch for the thesis program

### 7.2 Agent branches
- `agent/contract/protocol-lock`
- `agent/confirmatory/main-claim`
- `agent/falsification/negative-controls`
- `agent/robustness/split-strength`
- `agent/transfer/cross-person`
- `agent/interpretability/stability`
- `agent/synthesis/evidence-pack`
- `agent/quality/invariants`

Each agent works on exactly one branch.

No two agents may share a branch.

Each agent should preferably use a separate **git worktree** or fully separate checkout.

---

## 8. Merge Authority

There is exactly one merge authority: the **integration manager**.

Rules:
- no agent may merge directly into `main`
- no agent may merge directly into `program/thesis-evidence`
- all agent changes must enter through pull requests
- only the integration manager may approve and merge agent PRs
- `main` receives changes only from reviewed, stable integration states

Copilot, Codex, and other AI tools may help author code, but they are not merge authority.

---

## 9. Ownership Model

Agents may read any part of the repository.

Agents may write only within:
- their owned directories
- their owned manifests
- their owned tests/docs
- explicitly approved interface files

Agents must not perform opportunistic cross-family refactors.

If an agent needs a shared interface changed, it must open an **interface request** rather than editing another family’s area directly.

---

## 10. Non-Negotiable Rules

1. **No hidden scientific defaults**  
   Confirmatory science must be explicit in frozen contracts, not silently injected through adapters or helper behavior.

2. **No silent refactors**  
   No “while I was here” structure changes outside branch mission.

3. **No evidence-level drift**  
   Secondary, robustness, or diagnostic evidence may not be relabeled as confirmatory.

4. **No unofficial artifact in thesis outputs**  
   Thesis-ready synthesis must consume only official, promoted evidence.

5. **No branch overlap by habit**  
   Cross-owned edits require explicit approval.

6. **No overclaiming**  
   Repository outputs must remain consistent with the claim boundaries in this file.

7. **No engineering change that silently changes science**  
   Quality work may strengthen tests and invariants, but must not alter estimands or scientific assumptions without contract approval.

8. **No agent decides new science alone**  
   New evidence families or new claim types require program-level approval.

---

## 11. Wave Plan

### Wave 1 — Foundation
Agents:
- Contract
- Quality

Required outcomes:
- frozen program contract
- explicit evidence-family labels
- explicit ownership map
- CI/invariant protections in place
- no hidden scientific defaults in official paths

### Wave 2 — Core Science
Agents:
- Confirmatory
- Falsification
- Robustness
- Transfer
- Interpretability

Required outcomes:
- one narrow official main-claim path
- one adversarial control family
- one robustness family
- one transfer family
- one interpretation/stability family

### Wave 3 — Thesis Synthesis
Agents:
- Synthesis
- Quality final pass

Required outcomes:
- official evidence pack
- claim ledger
- decision map
- results-to-RQ mapping
- thesis-ready figure/table pack
- limitations register

---

## 12. Review Criteria for Every Agent PR

Every PR must be reviewed against the following questions:

1. Does it stay within the agent’s mission?
2. Does it stay within owned scope?
3. Does it change any scientific assumption?
4. If yes, is that change explicit and approved?
5. Does it preserve the evidence hierarchy?
6. Does it preserve claim boundaries?
7. Does it add or update the necessary tests?
8. Does it create hidden dependencies on another agent’s unfinished work?
9. Does it improve thesis completeness without increasing ambiguity?

A PR that is technically good but scientifically ambiguous should not be merged.

---

## 13. Required Program Artifacts

The final thesis program should be able to emit the following official artifacts:
- protocol lock record
- promoted confirmatory results
- promoted control/falsification results
- promoted robustness results
- promoted transfer results
- promoted interpretability/stability results
- claim ledger
- decision map
- evidence-to-RQ map
- figure pack
- table pack
- limitations register
- run audit / provenance record

---

## 14. Success Condition

The thesis program is considered operationally complete when:
- all evidence families exist
- all official science flows through explicit contracts
- parallel agents can work without undoing one another
- only promoted evidence enters thesis synthesis
- the repository can generate a thesis-ready evidence pack from official outputs
- the system is scientifically auditable and engineering-robust

---

## 15. Amendment Rule

This document may be amended only by deliberate program-level decision.

Any amendment must:
- be reviewed explicitly
- state what changed
- state why it changed
- state whether any prior evidence is invalidated or reclassified

No silent amendment is allowed.
