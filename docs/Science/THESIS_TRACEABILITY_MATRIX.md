# Thesis Traceability Matrix

## Purpose

This document maps the thesis problem, research questions, contribution claim,
method decisions, protocols, and expected result artifacts to the repository.
Its purpose is to make the project auditable as a thesis-oriented scientific
study rather than only as a software project.

The matrix is intentionally designed for the **pre-results phase**. It states
what each thesis-facing claim must be supported by, where that support should
come from in the repository, and where it is expected to appear in the thesis.
It does **not** assume that the claims are already supported. It defines the
required evidence path.

## How to Use This File

1. Keep the wording of the problem, research question, contribution claim, and
   scope aligned with the thesis.
2. For each row, replace placeholder protocol names, commands, config paths,
   run IDs, and artifact paths with the exact repository identifiers used in
   the locked analysis.
3. Update the `Status` field only when the required evidence actually exists.
4. If a thesis claim changes, update the related matrix rows before running new
   confirmatory experiments.
5. If a protocol changes in a thesis-relevant way, update both this file and
   `SCIENTIFIC_POSITIONING.md`.

## Status Vocabulary

- `planned`: intended but not yet locked or executed
- `locked`: protocol is frozen for confirmatory use
- `executed`: run completed and artifacts exist
- `written`: evidence has been incorporated into the thesis text
- `not supported`: executed evidence does not support the intended claim
- `inconclusive`: executed evidence is insufficient for a clear conclusion

## Identifier Conventions

Use stable IDs so the same scientific element can be tracked across planning,
execution, and writing.

- `P-*` = problem statements
- `RQ-*` = research questions
- `C-*` = contribution claims
- `S-*` = scope or non-claim boundaries
- `D-*` = method decisions
- `PROT-*` = protocols or experiment families
- `ART-*` = output artifacts
- `TH-*` = thesis section targets

---

## A. Problem, Research Questions, and Contribution

| ID | Thesis element | Locked wording / meaning | Required repository evidence | Expected artifact(s) | Thesis destination | Status |
|---|---|---|---|---|---|---|
| P-01 | Problem of general interest | Apparent success in repeated-session fMRI decoding can be scientifically misleading if evaluation does not adequately control leakage, repeated-measures dependence, confounds, and over-interpretation. | A repository-level scientific positioning note; explicit leakage-aware protocol rules; run outputs showing that claim-relevant splits and controls are defined before interpretation. | `docs/SCIENTIFIC_POSITIONING.md`; confirmatory protocol document; audit-ready run manifest. | TH-1.1 Introduction / Problem | planned |
| P-02 | Knowledge gap | Within-person held-out-session generalization, cross-person transfer, and interpretability evidence are scientifically distinct and should not be conflated. | Separate protocol definitions and separate reporting paths for within-person decoding, cross-person transfer, and interpretability/robustness analysis. | Distinct protocol files or sections; separate result folders; separate result tables. | TH-1.1 Problem and aim; TH-2.2 Related work; TH-3.5 Validation design | planned |
| RQ-00 | Main research question | Under strictly leakage-aware, session-held-out evaluation, to what extent do repeated-session fMRI beta-map representations support reliable within-person decoding of a three-class coarse-affect target, and to what extent is that learned mapping transferable across individuals under domain shift? | A locked confirmatory design that operationalizes target, split policy, model comparison, metrics, and transfer evaluation without post hoc changes. | Primary confirmatory protocol; primary summary table; chapter-ready result figure(s). | TH-1.1 Research questions; TH-4 Results; TH-5 Discussion | planned |
| RQ-01 | Within-person stability | Can a model trained on a subset of sessions from one participant predict the chosen target on held-out sessions from the same participant under session-held-out validation? | Session-held-out within-person protocol; leakage-safe preprocessing; performance against appropriate baselines or controls. | `ART-WP-01` participant-level performance tables; `ART-WP-02` aggregated summary table; `ART-WP-03` confusion summaries. | TH-4 primary results; TH-5 answer to RQ1 | planned |
| RQ-02 | Cross-person transfer | To what extent does a model trained on one individual generalize to the other when applied with frozen parameters, and how does this compare with a model trained directly on the target individual under the same validation logic? | Frozen transfer protocol; explicit no-retraining rule; separate reporting from within-person evaluation. | `ART-XFER-01` frozen transfer table; `ART-XFER-02` comparative table versus within-person reference. | TH-4 secondary results; TH-5 answer to RQ2 | planned |
| RQ-03 | Interpretability and robustness evidence | Which features drive prediction, and are importance patterns stable across held-out sessions and validation folds in a way that supports stable condition-related signal rather than session-specific artifacts? | Locked interpretability procedure tied to trained models; fold-aware or session-aware stability analysis; explicit interpretation limits. | `ART-INT-01` importance summaries; `ART-INT-02` stability diagnostics; `ART-INT-03` caution note on interpretation scope. | TH-4 interpretability results; TH-5 answer to RQ3 | planned |
| C-01 | Contribution statement | An empirically evaluated, leakage-aware modelling and validation framework for repeated-session fMRI decoding that separates within-person cross-session generalization from cross-person transfer and uses interpretability-driven robustness checks as supporting evidence rather than mere visualization. | All rows for RQ-01, RQ-02, and RQ-03 satisfied under one coherent locked design. | Combined confirmatory evidence package; final chapter linkage. | TH-1.1 Contribution and scope; TH-2.3 Novelty; TH-5 originality and significance | planned |
| C-02 | Novelty source | The novelty lies in the evaluation setting and evidence strategy rather than in proposing a new high-capacity architecture. | Repository documents showing that novelty is operationalized through target definition, split logic, controls, transfer separation, interpretability rules, and reproducibility. | Positioning file; protocol file; comparison frame document; method chapter text. | TH-1.1 Contribution; TH-2.3 Novelty; TH-3 Method | planned |
| C-03 | Evidence plan | Support requires: (i) above-baseline within-person decoding under held-out-session evaluation, (ii) separately reported cross-person transfer, and (iii) interpretability-based robustness evidence that is plausible and reasonably stable under conservative evaluation. | Explicit claim evaluation rules and artifact checklist; predeclared thresholds or decision logic where appropriate. | `docs/CLAIM_EVALUATION_RULES.md`; result package checklist; signed-off run manifest. | TH-1.1 Contribution; TH-4 Results; TH-5 Conclusions | planned |
| S-01 | Scope boundary | The dataset supports within-person stability analysis and cross-case transfer analysis within this dataset, but not broad population-level generalization. | Scope statement embedded in repo docs and repeated in result reporting templates. | `docs/DATASET_SCOPE_AND_GENERALIZATION.md`; result template limitations section. | TH-1.1 Delimitation; TH-5 Limitations | planned |
| S-02 | Non-claim boundary | The repository does not justify clinical, causal, unrestricted external-generalization, or direct neural-localization claims from model coefficients alone. | Use-and-misuse boundaries; interpretation template; discussion template. | Scope note; interpretation guardrail note. | TH-3.7 Ethics; TH-5 Discussion | planned |

---

## B. Decision-to-Evidence Map

| ID | Method decision | Locked thesis rationale | Repository implementation anchor | Must support | Expected artifact(s) | Status |
|---|---|---|---|---|---|---|
| D-01 | Prediction target | Use a locked three-class coarse-affect target because the thesis contribution depends on a clearly defined, thesis-consistent outcome rather than a moving target. | `<replace: target-definition file or config key>` | RQ-00, RQ-01, RQ-02 | Target schema file; label audit summary. | planned |
| D-02 | Sample unit | Treat each condition beta map as one ML sample after masking to in-brain voxels, consistent with the thesis method description. | `<replace: sample-construction script/path>` | RQ-00, RQ-01, RQ-02 | Sample index table; feature-generation log. | planned |
| D-03 | Primary validation setting | Use within-person held-out-session validation as the primary test of stable predictive signal. | `<replace: primary confirmatory protocol ID>` | RQ-01, C-01 | Participant-specific fold manifest; primary score tables. | planned |
| D-04 | Secondary transfer setting | Evaluate cross-person transfer separately using frozen parameters and no re-training. | `<replace: transfer protocol ID>` | RQ-02, C-01 | Frozen transfer results; comparison table. | planned |
| D-05 | Leakage control | Any supervised preprocessing, tuning, and feature selection must be fit on training data only within each split. | `<replace: pipeline implementation path>` | P-01, RQ-01, RQ-02, C-01 | Pipeline audit log; split-aware preprocessing log. | planned |
| D-06 | Baseline / control policy | Use appropriate baselines and permutation-style controls where applicable to distinguish real signal from chance or design artifacts. | `<replace: control protocol ID>` | RQ-01, C-03 | Baseline table; permutation summary; chance-reference note. | planned |
| D-07 | Metrics | Report accuracy, balanced accuracy, and macro-F1, with participant-level and aggregated summaries. | `<replace: metrics config or evaluator path>` | RQ-01, RQ-02 | Main performance tables; metric definition note. | planned |
| D-08 | Model comparison frame | Compare models under the same locked split logic and preprocessing discipline; do not treat architecture novelty as the main contribution. | `<replace: model registry or experiment suite>` | C-02 | Model comparison table; run manifest. | planned |
| D-09 | Interpretability role | Use interpretability analyses as supporting evidence about model behavior, not direct proof of neural localization or mechanism. | `<replace: interpretability protocol ID>` | RQ-03, S-02 | Importance summaries; stability plots; interpretation limitation note. | planned |
| D-10 | Reproducibility and auditability | Ensure explicit indexing, configuration logging, run manifests, and standardized outputs so the thesis claims can be audited. | `<replace: reproducibility doc/path>` | C-01, C-03 | Reproducibility checklist; environment snapshot; artifact manifest. | planned |
| D-11 | Ethical and interpretive caution | Treat conservative validation and careful scope limitation as ethical as well as methodological requirements. | `<replace: ethics/governance doc>` | S-01, S-02 | Ethics note; misuse-boundaries note. | planned |
| D-12 | AI and software transparency | Record the software stack and any AI-assisted coding/writing support that influenced the thesis work, together with verification responsibility. | `<replace: AI/tool disclosure doc>` | TH-3.6, TH-3.7, TH-5 reflection-related reporting | Tool disclosure file; AI usage log. | planned |

---

## C. Protocol-to-Artifact Map

This section is the operational backbone. Each protocol should produce a small,
explicit set of artifacts that can be lifted directly into the thesis.

| Protocol ID | Purpose | Minimum protocol rule | Required artifact(s) | Supports thesis elements | Status |
|---|---|---|---|---|---|
| PROT-WP-PRIMARY | Primary within-person confirmatory evaluation | Train and test must be separated at the session level; preprocessing and any tuning must remain within training data for each split. | `ART-WP-01` participant-level score table; `ART-WP-02` aggregated score table; `ART-WP-03` confusion patterns; `ART-WP-04` fold manifest. | RQ-00, RQ-01, C-01, C-03 | locked |
| PROT-XFER-FROZEN | Secondary cross-person transfer evaluation | Train on participant A and apply frozen model to participant B, and vice versa, with no re-training on the target participant. | `ART-XFER-01` A->B table; `ART-XFER-02` B->A table; `ART-XFER-03` comparison against within-person reference. | RQ-02, C-01, S-01 | locked |
| PROT-BASELINE-CONTROL | Chance-reference and control evaluation | Predeclare baseline and permutation-style controls consistent with the same split logic as the primary analysis. | `ART-CTRL-01` baseline results; `ART-CTRL-02` permutation summary; `ART-CTRL-03` interpretive note. | RQ-01, C-03 | locked |
| PROT-INTERP-STABILITY | Interpretability and robustness analysis | Compute model-behavior explanations only from locked models/runs and summarize whether patterns are directionally plausible and reasonably stable across held-out folds or sessions. | `ART-INT-01` global importance summary; `ART-INT-02` fold/session stability summary; `ART-INT-03` interpretation caveat note. | RQ-03, C-01, S-02 | locked |
| PROT-REPRO-AUDIT | Reproducibility and audit package | Every confirmatory run must emit configuration, dataset fingerprint, software/environment info, and artifact manifest. | `ART-AUD-01` run manifest; `ART-AUD-02` config snapshot; `ART-AUD-03` dataset fingerprint; `ART-AUD-04` environment snapshot. | C-01, C-03, TH-3.6, TH-5 limitations/reproducibility | locked |
| PROT-SCOPE-GUARDRAIL | Scope and non-claim enforcement | Result templates and discussion templates must preserve the thesis scope and non-claims. | `ART-SCOPE-01` reporting template; `ART-SCOPE-02` limitations template; `ART-SCOPE-03` non-claim checklist. | S-01, S-02, TH-5 Discussion | locked |

---

## D. Artifact-to-Thesis Writing Map

This section prevents a common failure mode: producing many outputs without a
clear path into the thesis.

| Artifact ID | What it contains | Preferred thesis use | Minimum write-up requirement | Status |
|---|---|---|---|---|
| ART-WP-01 | Participant-level within-person metrics by fold or held-out session set | Main result table in Chapter 4 or appendix-backed summary | State exactly which participant, split policy, metric set, and sample counts are represented. | planned |
| ART-WP-02 | Aggregated within-person performance summary | Main Chapter 4 summary table / figure | Interpret relative to baselines and uncertainty; do not overstate external generalization. | planned |
| ART-WP-03 | Confusion patterns | Chapter 4 support figure/table | Use only to clarify error structure, not as a substitute for main metrics. | planned |
| ART-XFER-01 / ART-XFER-02 | Frozen transfer performance in each direction | Chapter 4 secondary result table | Keep transfer separate from within-person results and state domain-shift conditions. | planned |
| ART-XFER-03 | Cross-person vs within-person comparison | Chapter 4 comparison paragraph and Chapter 5 interpretation | Explicitly state that this is cross-case evidence only, not population generalization. | planned |
| ART-CTRL-01 / ART-CTRL-02 | Baseline and permutation-style controls | Chapter 4 validity subsection | Use these artifacts to support the claim that above-chance performance is not a trivial artifact of the design. | planned |
| ART-INT-01 / ART-INT-02 | Importance summaries and stability diagnostics | Chapter 4 interpretability subsection | Describe them as supporting robustness evidence about model behavior, not as direct localization evidence. | planned |
| ART-AUD-01 to ART-AUD-04 | Reproducibility package | Chapter 3 implementation and Chapter 5 limitations/reproducibility | Use to justify trustworthiness, reproducibility, and method transparency. | planned |
| ART-SCOPE-01 to ART-SCOPE-03 | Scope and non-claim materials | Chapter 5 limitations and conclusion wording | Use to prevent accidental overclaiming in the thesis write-up. | planned |

---

## E. Chapter-Level Traceability

| Thesis section | What the section must answer or justify | Matrix rows that must be satisfied |
|---|---|---|
| TH-1.1 Introduction / Problem | What is the general-interest problem, why it matters, and how it relates to computer and systems sciences | P-01, P-02, S-01 |
| TH-1.1 Research questions | What is the single umbrella RQ and what are the tightly related sub-questions | RQ-00, RQ-01, RQ-02, RQ-03 |
| TH-1.1 Contribution and scope | What is new, what evidence is required, and what remains out of scope | C-01, C-02, C-03, S-01, S-02 |
| TH-2 Background / Related work | Which prior work justifies the target, validation logic, transfer framing, and interpretation limits | D-01, D-03, D-04, D-05, D-06, D-09 |
| TH-3 Method / Research strategy | Why experiments are the chosen methodology and how they are operationalized | D-02, D-03, D-04, D-05, D-06, D-07, D-08 |
| TH-3 Method / Tools and ethics | Which tools were used, how reproducibility is supported, and what ethical guardrails apply | D-10, D-11, D-12 |
| TH-4 Results | What the executed evidence shows for within-person decoding, cross-person transfer, and interpretability/robustness | PROT-WP-PRIMARY, PROT-XFER-FROZEN, PROT-BASELINE-CONTROL, PROT-INTERP-STABILITY |
| TH-5 Discussion / Conclusions | Which RQs were answered, what limits apply, what the originality is, and what the results do **not** justify | C-01, C-03, S-01, S-02, ART-AUD-01 to ART-AUD-04 |

---

## F. Pre-Results Completion Checklist

The repository can be considered **traceability-ready before final results**
only if all items below are true.

- [ ] The wording of the main research question matches the thesis.
- [ ] RQ1, RQ2, and RQ3 are clearly separated in both documentation and protocol logic.
- [ ] The contribution statement is written exactly once and reused consistently.
- [ ] Scope limits and non-claims are explicit.
- [ ] The primary within-person protocol is locked.
- [ ] The frozen cross-person transfer protocol is locked.
- [ ] Baseline/control policy is defined.
- [ ] Interpretability procedure and its limits are defined.
- [ ] Reproducibility and artifact logging requirements are defined.
- [ ] AI/software transparency documentation exists.
- [ ] Placeholder repository paths in this file have been replaced with exact identifiers.

## G. Post-Execution Completion Checklist

After running the confirmatory analyses, update this file only if the evidence
actually exists.

- [ ] Replace `planned` with `locked`, `executed`, `written`, `not supported`, or `inconclusive` as appropriate.
- [ ] Add exact run IDs and artifact paths for each protocol.
- [ ] Record whether RQ1 was supported, not supported, or inconclusive.
- [ ] Record whether RQ2 was supported, not supported, or inconclusive.
- [ ] Record whether RQ3 yielded supportive robustness evidence, non-supportive evidence, or inconclusive evidence.
- [ ] Update the thesis writing plan so each Chapter 4 and Chapter 5 statement points to a concrete artifact.
- [ ] Do not upgrade any scientific claim unless the corresponding artifact exists and is interpretable under the declared scope.

## H. Maintenance Rule

Update this matrix whenever any of the following changes:

- the problem statement changes;
- the umbrella research question or any sub-question changes;
- the contribution statement changes;
- the target definition changes;
- the primary or secondary validation design changes;
- the interpretation scope changes;
- the location or identifier of a protocol or artifact changes in a way that affects auditability.

Do **not** change completed rows merely to make the thesis look cleaner after
results. If the executed evidence is mixed, negative, or inconclusive, the
matrix should preserve that fact.
