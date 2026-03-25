# Scientific Positioning

## Purpose

This document explains the scientific problem that this repository addresses,
how the work is positioned within computer and systems sciences, what the main
research question is, what contribution the project is intended to make, and
which claims are explicitly out of scope.

It is a thesis-facing positioning note for the repository. It complements the
execution and governance documents by making the scientific intent of the
project easy to find and hard to misstate.

The protocol files and run artifacts remain the operational source of truth for
executed experiments. This document states the scientific framing that those
artifacts are expected to serve.

## One-Sentence Positioning

This repository supports a methodological study in computer and systems
sciences on how to perform trustworthy, leakage-aware machine-learning
evaluation for repeated-session fMRI decoding, with primary emphasis on
within-person cross-session generalization and secondary analysis of
cross-person transfer under domain shift.

## Problem of General Interest

The problem underlying this work is that apparent success in neuroimaging-based
emotion decoding can be scientifically misleading when repeated-measures data
are evaluated without sufficiently strict controls for leakage, dependence,
confounds, and over-interpretation.

In repeated-session fMRI, samples are not independent in the simple sense often
assumed by generic machine-learning workflows. If train-test splits, feature
construction, preprocessing, tuning, or interpretation procedures are not
aligned with the nested structure of subject, session, run, and condition,
performance estimates may look stronger than the underlying evidence actually
supports.

The scientific problem is therefore not only whether affect-related information
can be predicted from fMRI beta maps, but whether such prediction reflects
stable condition-related signal under conservative evaluation rather than
session-specific artifacts, leakage, or other non-target structure.

## Knowledge Gap

Current decoding literature often reports predictive performance, but the
strength of the scientific claim depends heavily on the evaluation setting.
Repeated-session neuroimaging introduces a specific methodological gap:

1. within-person cross-session generalization and cross-person transfer are
   scientifically different questions and should not be conflated;
2. repeated-measures data require split definitions and preprocessing policies
   that are explicitly leakage-aware;
3. interpretability outputs are often shown, but they are not always used under
   a disciplined evidence model that distinguishes supportive robustness
   evidence from strong claims about neural localization or mechanism.

This repository is designed around that gap. Its purpose is not to maximize a
single benchmark score in the abstract. Its purpose is to support a decoding
study in which target definition, split logic, model comparison, controls,
reproducibility, and interpretability are aligned with the scientific risks of
repeated-session fMRI analysis.

## Research Question

The main research question supported by this repository is:

> Under strictly leakage-aware, session-held-out evaluation, to what extent do
> repeated-session fMRI beta-map representations support reliable within-person
> decoding of a three-class coarse-affect target, and to what extent is the
> learned mapping transferable across individuals under domain shift?

The project further separates this question into three tightly related parts:

- **Within-person stability:** Can a model trained on a subset of sessions from
  one participant predict the target on held-out sessions from the same
  participant?
- **Cross-person transfer:** Does a model trained on one participant transfer to
  the other participant when applied with frozen parameters and no re-training?
- **Interpretability and robustness evidence:** Do the learned importance
  patterns show recurring and at least partially stable explanatory structure
  across held-out sessions and validation folds?

## Field Positioning in Computer and Systems Sciences

This work contributes to **computer and systems sciences**, specifically to
**trustworthy machine learning for high-dimensional repeated-measures
scientific data**.

More concretely, the project is positioned at the intersection of:

- machine-learning evaluation methodology;
- robust and reproducible computational experimentation;
- leakage-aware modelling under hierarchical repeated-measures data;
- interpretation-aware analysis of predictive models in scientific settings.

The contribution is therefore primarily methodological. The repository is not
positioned as a new clinical system, a production decision-support tool, or a
claim about universal emotion representation. It is positioned as an auditable
framework for structuring, validating, and interpreting repeated-session fMRI
decoding experiments in a scientifically defensible way.

## Contribution Statement

This project supports the following contribution claim:

> An empirically evaluated, leakage-aware modelling and validation framework for
> repeated-session fMRI decoding that separates within-person cross-session
> generalization from cross-person transfer and treats interpretability-driven
> analyses as supporting robustness evidence rather than mere visualization.

### Novelty source

The novelty lies primarily in the **evaluation and evidence strategy**, not in a
claim of architectural novelty.

In particular, the project combines:

- held-out-session within-person evaluation as the primary setting;
- separate cross-person transfer analysis under domain shift;
- reproducibility and governance controls suitable for frozen confirmatory runs;
- interpretability analyses used under explicit scope limits and evidence
  boundaries.

### Comparison frame

The project improves on less disciplined decoding setups in which repeated
samples are evaluated with insufficient separation between train and test,
within-person and cross-person evidence are mixed together, or explanatory maps
are presented without clear limits on what they justify.

### Evidence plan

The contribution is supported only if the final thesis evidence shows all of the
following under the locked design:

1. within-person decoding performs above appropriate baselines under
   held-out-session evaluation;
2. cross-person transfer is reported separately and interpreted more cautiously
   than within-person generalization;
3. interpretability-based analyses provide supportive robustness evidence that
   is plausible and reasonably stable under conservative validation;
4. the full analysis remains reproducible, auditable, and consistent with the
   declared protocol and governance boundaries.

## Delimitations and Non-Claims

This repository is intentionally scoped. The following boundaries are part of
its scientific framing, not retrospective disclaimers.

### Delimitations

- The current thesis setting uses a repeated-session dataset with two
  participants.
- The primary target is a locked three-class coarse-affect formulation.
- The primary scientific setting is **within-person held-out-session**
  generalization.
- Cross-person transfer is a **secondary** and distinct analysis setting.
- The study evaluates predictive association under a specific task formulation
  and validation design.

### Non-claims

This repository does **not** by itself justify claims about:

- clinical diagnosis, treatment selection, or deployment readiness;
- causal inference from predictive performance;
- unrestricted population-level external generalization;
- direct neural localization from model coefficients alone;
- universal, context-free, or mind-reading interpretations of emotion.

## Why This Framing Fits the Repository

The repository structure already reflects this scientific positioning.

- Frozen protocol execution supports confirmatory thesis evidence rather than
  ad hoc post hoc reporting.
- Reproducibility and artifact verification support auditability.
- Governance documents separate intended use from misuse and overclaiming.
- Mode separation distinguishes exploratory work from locked comparisons and
  confirmatory analysis.
- Leakage-aware split logic is treated as a scientific requirement, not only as
  a software preference.

In short, the repository is designed to make the scientific claim narrower,
clearer, and more trustworthy.

## Relationship to Other Repository Documents

This document should be read together with the following files:

- [`README.md`](../README.md): repository overview and execution entry points.
- [`docs/EXPERIMENTS.md`](./EXPERIMENTS.md): framework modes and execution
  intent.
- [`docs/confirmatory/confirmatory_plan_v1.md`](./confirmatory/confirmatory_plan_v1.md):
  locked confirmatory study plan.
- [`docs/USE_AND_MISUSE_BOUNDARIES.md`](./USE_AND_MISUSE_BOUNDARIES.md):
  intended use, misuse boundaries, and scope limits.
- [`docs/CONFIRMATORY_READY.md`](./CONFIRMATORY_READY.md): release-facing
  confirmatory criteria.
- [`docs/REPRODUCIBILITY.md`](./REPRODUCIBILITY.md): reproducibility workflow.

If there is a conflict between this document and a locked protocol used for an
official thesis run, the protocol governs the executed analysis and this file
should be updated accordingly.

## Maintenance Rule

Update this document only when at least one of the following changes:

- the main research question changes;
- the primary target definition changes;
- the primary evaluation setting changes;
- the contribution claim changes;
- the scope boundaries or non-claims change in a thesis-relevant way.

Minor implementation changes that do not alter the scientific framing should
not trigger edits here.

## Status

Current status: suitable as a repository-level scientific positioning statement
for the thesis project before final results are produced.
