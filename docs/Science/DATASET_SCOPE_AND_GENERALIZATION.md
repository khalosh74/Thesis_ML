# Dataset Scope and Generalization

## Purpose

This document defines what the current dataset can and cannot support as
scientific evidence, how generalization should be described, and which claim
boundaries must remain in force when repository outputs are translated into
thesis text, figures, tables, or future reports.

It is a scope-control document. Its purpose is not to justify the entire study
again, but to prevent the dataset from being interpreted more broadly than the
design and evidence allow.

This file should be read together with:

- `docs/SCIENTIFIC_POSITIONING.md`
- `docs/THESIS_TRACEABILITY_MATRIX.md`
- `docs/CLAIM_EVALUATION_RULES.md`
- `docs/confirmatory/confirmatory_plan_v1.md`
- `docs/USE_AND_MISUSE_BOUNDARIES.md`

## One-sentence rule

The current dataset supports an **individual-level methodological study** of
trustworthy repeated-session fMRI decoding, with **within-person held-out-session
generalization** as the primary target and **cross-person transfer as secondary
cross-case evidence**, but it does **not** support broad population-level
external generalization claims.

## Dataset facts relevant to scope

The current thesis setting uses a repeated-session precision-neuroimaging
dataset with the following properties:

- two participants;
- repeated scanning sessions per participant;
- first-level GLM-derived fMRI beta-map representations;
- a locked emotion-to-`coarse_affect` target mapping for the confirmatory
  thesis setting;
- hierarchical dependence across participant, session, run, and sample.

This document governs scientific interpretation of those properties. It does
not replace the exact implementation contract in the locked protocol.

## Why this dataset is scientifically meaningful

A small number of participants does not make the dataset scientifically
meaningless. In this thesis, the dataset is meaningful because the primary
scientific target is **stability and generalization within the individual under
repeated measurement**, not estimation of a population average effect.

That distinction matters. In affective neuroscience and dense-sampling
neuroimaging, repeated measurement can be scientifically preferable when the
underlying phenomenon is strongly individual and when the first question is
whether a signal is real, stable, and reproducible at the level where it occurs.
For the present repository, this supports using repeated sessions to test whether
models generalize to **held-out sessions from the same participant** under a
leakage-aware design.

Accordingly, the dataset is fit for a **methodological contribution** about how
to structure, evaluate, and interpret repeated-session fMRI decoding under
strict claim boundaries. It is not fit for unrestricted claims about how people
in general would perform under the same task.

## Primary unit of inference

The primary unit of inference in this repository is the **individual under
repeated measurement**, not the population average.

This means the repository is optimized to answer questions such as:

- Does a model trained on some sessions from a participant predict held-out
  sessions from that same participant?
- Does the full workflow remain leakage-aware under hierarchical repeated
  measures?
- Are any supportive explanation patterns at least somewhat recurring across
  held-out folds or sessions?

This repository is not optimized to estimate population prevalence, average
clinical performance, or universal affect representations.

## Levels of generalization

The study contains multiple generalization levels. They must not be conflated.

| Generalization level | What it asks | Status in this repository | Allowed interpretation |
|---|---|---|---|
| Within-person held-out-session generalization | Does a model trained on some sessions predict unseen sessions from the same participant? | Primary | Internal evidence about stability and predictive usability within the same individual under the locked design |
| Cross-person frozen transfer | Does a model trained on one participant transfer to the other without re-training? | Secondary | Cross-case transfer evidence under domain shift |
| External-dataset compatibility | Does another dataset appear structurally compatible with the pipeline and declared target assumptions? | Compatibility-only in this phase | Governance and feasibility evidence only, not external performance |
| Population-level generalization | Would the same result hold broadly in a target population? | Not established by the current dataset | Out of scope |

## What the dataset does support

### 1. Internal methodological evaluation

The dataset supports evaluating whether a leakage-aware repeated-session fMRI
pipeline can produce credible internal evidence under the locked protocol.

This includes:

- dataset-contract checks and fingerprinting;
- leakage-aware session-held-out evaluation;
- dummy-baseline and permutation-based controls;
- auditable fold-level predictions and split logs;
- disciplined interpretation rules tied to artifacts.

### 2. Within-person stability as the primary scientific target

The dataset supports asking whether condition-level or event-level fMRI beta-map
representations remain sufficiently stable within a participant to support
prediction across held-out sessions.

This is the primary scientific question because repeated sessions provide the
temporal separation needed to distinguish easier in-sample fitting from the more
meaningful question of future-session generalization within the same person.

### 3. Cross-case transfer under domain shift

The dataset supports a narrow secondary question: whether a model trained on one
participant can transfer to the other under frozen application and no
re-training.

This is scientifically meaningful, but it must be interpreted as a **harder and
narrower transfer test** than within-person prediction. It provides evidence
about transfer **between these cases** under domain shift, not about broad human
population performance.

### 4. Failure-mode analysis

The dataset supports studying scientific risks such as:

- leakage from repeated-measures dependence;
- inflated performance from weak controls;
- confusion between within-person and cross-person evidence;
- over-interpretation of model explanations;
- narrative overclaiming relative to the actual evidence.

That capability is part of the thesis contribution. A trustworthy result in this
repository depends not only on performance values, but also on whether these
risks were controlled and reported correctly.

## What the dataset does not support

The current dataset does **not** support the following claims.

### 1. Broad population-level external generalization

Two participants are not sufficient to claim that a decoder generalizes to a
population, even if internal results are strong. At most, the current dataset
supports participant-specific evidence plus cross-case transfer between the
observed cases.

### 2. Clinical effectiveness or deployment readiness

The repository does not establish that a model is suitable for diagnosis,
intervention, triage, or any real-world decision support context.

### 3. Causal conclusions

Predictive performance does not establish that the decoded features are causal
mechanisms of emotion, nor that manipulating the identified patterns would
change the target.

### 4. Anatomical localization from model coefficients alone

Interpretability outputs in this repository are supporting evidence about model
behavior. They are not direct evidence that the highest-weighted voxels are the
unique neural generators of the target state.

### 5. Universal or context-free emotion representation

The current dataset does not justify a claim that the learned signal represents
a universal, context-independent, or exhaustive neural code for emotion.

## Required reporting rules

Any thesis chapter, paper draft, slide deck, or repository report using this
dataset must follow these rules.

### Report participant-level evidence explicitly

Results should be reported per participant whenever the claim concerns
within-person generalization. Aggregate summaries may be shown, but they must
not hide divergence between participants.

### Keep primary and secondary settings separate

Within-person held-out-session results and cross-person frozen-transfer results
must be reported in separate subsections, tables, or clearly labeled panels.
They answer different scientific questions.

### State the generalization target in every important result paragraph

Each major result paragraph should make clear whether the evidence concerns:

- held-out sessions from the same participant;
- transfer to the other participant;
- compatibility-only external checks; or
- future work beyond the current evidence.

### Treat external compatibility as governance evidence only

If compatibility artifacts are produced for an external dataset, they must not
be described as external validation performance. They show only whether a future
external validation step is structurally feasible.

### Keep limitations attached to the claim

The limitation should appear where the claim appears. Do not state a strong
result in one paragraph and defer the scope boundary to a distant limitation
section.

## Approved interpretation patterns

The following kinds of statements are allowed, assuming the corresponding
artifacts satisfy the locked rules.

### Within-person primary claim

- "Under held-out-session evaluation, the model showed evidence of within-person
  decoding for the declared target within the locked dataset."
- "This supports the presence of predictive information that remains usable
  across unseen sessions from the same participant under the declared protocol."
- "The result is internal and individual-level; it does not establish population
  generalization."

### Cross-person secondary claim

- "Frozen cross-person transfer provided secondary cross-case evidence under
  domain shift."
- "Transfer performance was weaker than within-person performance, which is
  consistent with the greater difficulty of across-person generalization."
- "This result should be interpreted as evidence about these cases, not as
  population-level external validity."

### Methodological contribution claim

- "The study supports a methodological contribution about how repeated-session
  neuroimaging decoding can be evaluated and interpreted more trustworthily."
- "The contribution lies in the alignment of target definition, split logic,
  controls, auditability, and claim boundaries."

## Prohibited or misleading interpretation patterns

Do **not** use statements of the following kind.

- "The model generalizes to people in general."
- "The model demonstrates population-level emotion decoding."
- "The findings validate a clinically useful biomarker."
- "The explanation maps show where the emotion is located in the brain."
- "Cross-person success proves external generalization."
- "Because the internal result is strong, the small number of participants no
  longer matters."
- "An external compatibility report demonstrates external performance."

## How to handle strong versus weak results

### If results are strong

Strong internal results still remain internal results. The wording may become
more confident about the **quality of evidence within scope**, but the scope
itself must not expand.

### If results are weak or mixed

Weak or mixed results do not mean the study failed. They may still support a
useful methodological conclusion, for example that the stricter evaluation goal
is demanding, that transfer is more fragile than within-person decoding, or
that some apparent effects do not survive conservative controls.

### If results differ across participants

Participant heterogeneity must be treated as scientifically informative rather
than as an inconvenience. With repeated-measures individual-level data, such
heterogeneity is part of the phenomenon and should be reported directly.

## External validation and future extension

A future study may extend beyond the current scope, but that requires new
evidence rather than new wording.

To support stronger generalization claims, future work would need one or more of
the following:

- additional participants sampled to address the intended target population;
- a predeclared external validation dataset with true out-of-dataset
  performance analysis;
- harmonized task, label, preprocessing, and feature-space assumptions across
  datasets;
- explicit protocol updates that promote external validation from
  compatibility-only status to claim-bearing evidence.

Until then, external generalization remains out of scope for the current thesis
repository.

## Practical rule for thesis writing

When writing a result or conclusion from this repository, always complete the
following sentence mentally before finalizing the wording:

> This claim generalizes to **which unit**, under **which split**, for **which
> target**, and with **which explicit non-claims**?

If that sentence cannot be answered precisely, the wording is too broad.

## Relationship to thesis grading and scientific framing

This document helps preserve alignment with the thesis framing by ensuring that:

- the problem and contribution remain easy to identify;
- the methodological contribution stays within computer and systems sciences;
- the discussion of validity, reliability, transferability, and generalizability
  remains explicit;
- conclusions stay proportional to the available evidence.

## Precedence rule

If this document conflicts with a locked confirmatory protocol or explicit
repository misuse boundary, use the following order of precedence:

1. `docs/confirmatory/confirmatory_plan_v1.md`
2. `docs/USE_AND_MISUSE_BOUNDARIES.md`
3. `docs/CLAIM_EVALUATION_RULES.md`
4. `docs/SCIENTIFIC_POSITIONING.md`
5. `docs/DATASET_SCOPE_AND_GENERALIZATION.md`
6. thesis chapter text

## Source basis

This document is derived from the locked scientific framing already established
for the thesis and repository, especially:

- the thesis statement that the dataset contains two participants with repeated
  sessions and supports analysis of within-person stability and cross-case
  transfer within this dataset, but not broad population-level generalization;
- the thesis distinction between within-person cross-session generalization as
  the primary setting and cross-person transfer as a separate, more demanding
  domain-shift problem;
- the research-standard requirement that, with two individuals, no population
  generalizations are made and cross-person findings are treated as cross-case
  evidence;
- the Single-N rationale that individual-level repeated measurement can be
  scientifically appropriate when the aim is to establish real and stable
  effects at the level of the individual before broader generalization claims
  are attempted.
