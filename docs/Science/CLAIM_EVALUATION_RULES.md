# Claim Evaluation Rules

## Purpose

This document defines how thesis-facing claims are evaluated after results are
produced. Its purpose is to prevent post hoc claim inflation and to ensure that
thesis wording follows the locked protocol, the repository governance rules,
and the actual evidence emitted by confirmatory runs.

This file is not a replacement for the protocol. The protocol defines what is
run. This file defines how the resulting evidence may and may not be turned into
scientific claims in the thesis.

## Scope

These rules apply to four claim layers:

1. the primary confirmatory claim about within-person held-out-session decoding;
2. the secondary claim about frozen cross-person transfer;
3. the supporting claim about interpretability and robustness evidence;
4. the overall thesis contribution claim.

The rules are written for the current thesis framing:

- primary setting: within-person held-out-session evaluation;
- primary metric: `balanced_accuracy`;
- required controls: dummy baseline and label permutation test;
- secondary setting: frozen cross-person transfer;
- interpretation scope: predictive association under internal validation, not
  causation, clinical use, localization, or unrestricted external
  generalization.

## Governing documents and precedence

When sources disagree, use the following order of precedence:

1. `docs/confirmatory/confirmatory_plan_v1.md`
2. `docs/CONFIRMATORY_READY.md`
3. `docs/USE_AND_MISUSE_BOUNDARIES.md`
4. `docs/REPRODUCIBILITY.md`
5. `docs/SCIENTIFIC_POSITIONING.md`
6. `docs/THESIS_TRACEABILITY_MATRIX.md`
7. thesis chapter text

If a future locked protocol replaces `confirmatory_plan_v1`, update this file
so that the rule set matches the new locked source.

## Core principles

### Locked protocol first

No result may be used to support a confirmatory claim unless it was produced
under the locked target definition, locked split logic, locked model policy,
and locked metric policy.

### No claim without an artifact

A thesis claim must point to a concrete result artifact, not just a memory of a
run or an intermediate console output.

### Controls are mandatory, not decorative

For the primary confirmatory claim, dummy-baseline performance and
permutation-based evidence are required parts of the decision rule.

### Secondary evidence cannot rescue primary failure

Subgroup analyses, exploratory analyses, transfer analyses, or attractive
interpretability figures may not be used to rescue a failed primary
confirmatory result.

### Negative results are valid results

A rigorous negative or mixed result is scientifically valid. It must be written
up honestly and must not be reframed as support for a stronger claim than the
evidence allows.

### Scope limits remain in force even when results are strong

Strong internal performance does not justify claims about causation, clinical
readiness, neural localization from coefficients alone, or broad population
external generalization.

## Allowed status labels

Use the status labels below consistently.

| Status | Meaning | Allowed for |
|---|---|---|
| `supported` | The required evidence exists and satisfies the predeclared rule set. | primary claim, interpretability claim, overall contribution |
| `partially_supported` | Some but not all required evidence pillars support the broader thesis-level claim. Use this only when the evidence is mixed but still meaningful under the declared scope. | secondary claim, interpretability claim, overall contribution |
| `not_supported` | The required evidence exists and does not satisfy the rule set. | all claim layers |
| `inconclusive` | The evaluation cannot be judged fairly because required evidence is missing, incomplete, or uninterpretable, but no confirmed protocol violation has been established. | all claim layers |
| `invalid_for_confirmatory_interpretation` | The run or artifact violates confirmatory requirements or lacks the minimum governance conditions required for confirmatory use. | all claim layers |

### Important usage note

For the primary confirmatory claim, the preferred final statuses are usually:

- `supported`
- `not_supported`
- `invalid_for_confirmatory_interpretation`

Use `inconclusive` for the primary claim only when the necessary evidence is
missing or unusable, not as a softer wording for a failed result.

## Required evidence bundle

Before any claim is classified, confirm that the minimum evidence bundle exists.

### Confirmatory validity bundle

All of the following must be present for confirmatory interpretation:

- official artifact verification passes;
- `confirmatory_status = confirmatory`;
- no science-critical deviation is detected;
- `controls_valid_for_confirmatory = true`;
- `required_evidence_status.valid = true`;
- dataset fingerprint evidence is present;
- all confirmatory runs are completed;
- if a reproducibility summary is provided, it reports `passed = true`.

### Primary analysis bundle

All of the following must be present for the primary claim:

- the locked `within_subject_loso_session` evaluation;
- primary metric output for `balanced_accuracy`;
- dummy baseline under the same split structure;
- label permutation test under the locked structure;
- fold-level predictions;
- fold-level split log;
- run configuration snapshot and protocol ID;
- sufficient reporting to identify participant-level and aggregated results.

### Secondary analysis bundle

All of the following must be present for the transfer claim:

- frozen cross-person transfer outputs in the declared direction(s);
- explicit confirmation that no re-training occurred on the target person;
- the same locked target definition as the primary analysis;
- descriptive comparison against the within-person reference;
- explicit thesis wording that transfer is secondary and cross-case only.

### Interpretability bundle

All of the following must be present for the interpretability claim:

- explanation outputs derived from locked model runs;
- a documented explanation method;
- summaries across held-out folds and/or sessions;
- a caveat note describing what the explanation can and cannot support.

## Rule set for the primary confirmatory claim

### Claim being evaluated

> Under the locked within-subject held-out-session protocol, a fixed linear
> classifier predicts `coarse_affect` above chance on held-out sessions.

### Classify as `supported` only if all conditions below are true

1. the confirmatory validity bundle passes;
2. the primary analysis bundle is complete;
3. the locked primary metric is `balanced_accuracy`;
4. observed primary performance exceeds the dummy baseline under the same split
   structure;
5. the result survives the locked permutation-based decision rule;
6. the thesis report states the result as internal, leakage-aware,
   held-out-session evidence only;
7. no prohibited interpretation is attached to the result.

### Classify as `not_supported` if any condition below is true while the run remains otherwise valid

- primary performance does not exceed the dummy baseline;
- primary performance fails the permutation-based decision rule;
- the result is validly executed but does not support above-chance decoding
  under the locked analysis.

### Classify as `inconclusive` only if one of the following applies

- the run appears valid but one or more required artifacts are missing;
- the required control outputs are incomplete or unreadable;
- the result cannot be interpreted because the evidence bundle is materially
  incomplete, yet no confirmed protocol violation has been established.

### Classify as `invalid_for_confirmatory_interpretation` if any of the following applies

- confirmatory-ready criteria fail;
- science-critical deviations are present;
- target, split, metric, or model policy changed after lock;
- controls required for confirmatory use are absent or invalid;
- evidence needed to audit the run is missing in a way that prevents
  governance verification.

### Additional rule for participant heterogeneity

Participant-level heterogeneity must be reported explicitly. It does not
automatically nullify a supported aggregate result, but it must reduce the
strength of the thesis narrative whenever support depends heavily on one
participant or when the effect is not directionally consistent across
participants.

Do not hide participant-level divergence behind aggregate reporting.

## Rule set for the frozen cross-person transfer claim

### Claim being evaluated

> A model trained on one participant transfers to the other participant under
> frozen application and no re-training.

This is a **secondary** claim. It is not primary confirmatory evidence unless a
future protocol explicitly promotes it.

### Classify as `supported` only in the narrow secondary sense

Use `supported` for this claim only when all of the following are true:

1. the secondary analysis bundle is complete;
2. transfer was performed with frozen parameters and no target-person
   re-training;
3. transfer performance is above the relevant control reference;
4. the result is written as cross-case transfer evidence under domain shift,
   not as external population generalization.

### Classify as `partially_supported` if any of the following applies

- only one transfer direction shows supportive evidence;
- transfer is above baseline but much weaker than within-person performance and
  clearly unstable;
- evidence exists but is mixed enough that only cautious descriptive support is
  justified.

### Classify as `not_supported` if any of the following applies

- transfer is at or below the relevant control reference;
- transfer performance is too weak to justify even cautious descriptive support;
- the result contradicts the idea of useful frozen transfer under the locked
  setting.

### Classify as `inconclusive` if any of the following applies

- transfer artifacts are incomplete;
- one or both required directions are missing and no justified locked reason is
  recorded;
- evidence exists but cannot be interpreted fairly.

### Classify as `invalid_for_confirmatory_interpretation` if any of the following applies

- any re-training, tuning, or leakage occurred on the target participant;
- the transfer result was produced outside the locked target/scope definition;
- the analysis is presented as confirmatory even though the protocol defines it
  as secondary descriptive evidence only.

## Rule set for the interpretability and robustness claim

### Claim being evaluated

> Interpretability outputs provide supporting robustness evidence about model
> behavior under conservative evaluation.

This claim is intentionally narrower than a localization or mechanism claim.

### Classify as `supported` only if all of the following are true

1. the interpretability bundle is complete;
2. explanations are derived from locked model runs rather than from ad hoc
   exploratory reruns;
3. patterns show at least some recurring structure across held-out folds and/or
   sessions;
4. the patterns are plausible relative to the task and target definition;
5. the thesis text states that the explanations are supporting evidence about
   model behavior, not direct proof of anatomical localization or causal
   mechanism.

### Classify as `partially_supported` if any of the following applies

- explanation outputs exist and are plausible, but stability is weak or mixed;
- some folds or sessions agree while others do not;
- the evidence is useful for cautious interpretation but too unstable for a
  stronger robustness statement.

### Classify as `not_supported` if any of the following applies

- explanation patterns are too unstable to support meaningful robustness
  interpretation;
- the explanation is dominated by clearly implausible or non-task-consistent
  structure;
- the evidence does not add credible support beyond the predictive result.

### Classify as `inconclusive` if any of the following applies

- explanation outputs are missing or incomplete;
- aggregation or stability summaries are absent;
- a fair interpretation cannot be made from the available artifacts.

### Classify as `invalid_for_confirmatory_interpretation` if any of the following applies

- explanation outputs were generated from non-locked models but presented as if
  they were part of the confirmatory evidence;
- explanation results are used to justify claims that exceed the declared scope,
  such as neural localization or causal interpretation.

## Rule set for the overall thesis contribution claim

### Claim being evaluated

> This thesis contributes a leakage-aware and interpretable evaluation
> framework for repeated-session fMRI decoding that separates within-person
> generalization from cross-person transfer and uses interpretability as
> supporting robustness evidence.

### Classify as `supported` only if all of the following are true

1. the primary confirmatory claim is `supported`;
2. cross-person transfer is reported separately and interpreted within scope,
   regardless of whether it is strong or weak;
3. interpretability is at least `supported` or `partially_supported` as a
   supporting evidence layer;
4. reproducibility and auditability evidence are present;
5. the final thesis preserves all non-claims and scope boundaries.

### Classify as `partially_supported` if any of the following applies

- the primary confirmatory claim is supported, but one supporting pillar is
  weak, mixed, or incomplete;
- the workflow and governance contribution are demonstrated, but transfer or
  interpretability evidence is limited;
- the primary confirmatory claim is `not_supported`, but the thesis is
  explicitly narrowed to a rigorous negative methodological finding and does
  not present the negative result as successful decoding;
- the thesis yields a valid but narrower methodological contribution than the
  strongest intended version.

### Classify as `not_supported` if any of the following applies

- the confirmatory evidence cannot support even a narrowed methodological
  contribution under honest scope limits;
- the final thesis presents a failed or invalid primary result as if it were
  successful confirmatory support;
- the final thesis overclaims beyond what the evidence justifies.

### Classify as `inconclusive` if any of the following applies

- confirmatory execution is incomplete;
- required artifacts for judging the contribution are missing;
- the evidence base is too incomplete to determine whether the contribution was
  realized.

### Classify as `invalid_for_confirmatory_interpretation` if any of the following applies

- the thesis-level conclusion relies on invalid runs or uncontrolled analyses;
- the reported conclusion conflicts with the protocol/governance state of the
  executed evidence.

## Approved writing patterns by status

Use these as wording templates in Chapter 4 and Chapter 5.

### Primary claim supported

Use wording of this form:

> Under the locked within-subject held-out-session protocol, the confirmatory
> analysis found above-baseline predictive performance for `coarse_affect` on
> held-out sessions. This supports within-person decoding under the declared
> internal evaluation setting.

### Primary claim not supported

Use wording of this form:

> Under the locked within-subject held-out-session protocol, the confirmatory
> analysis did not provide evidence that `coarse_affect` could be decoded above
> the required control threshold on held-out sessions.

### Primary claim invalid

Use wording of this form:

> This result cannot be used as confirmatory evidence because the required
> confirmatory conditions were not satisfied.

### Cross-person transfer supported or partially supported

Use wording of this form:

> Frozen cross-person transfer showed [supportive / mixed] secondary evidence
> under domain shift. This result is interpreted as cross-case transfer only
> and does not establish population-level external generalization.

### Interpretability claim supported or partially supported

Use wording of this form:

> Interpretability outputs provided [supportive / mixed] evidence about model
> behavior across held-out evaluations. These patterns are used as supporting
> robustness evidence and not as direct localization claims.

### Overall contribution partially supported after a rigorous negative result

Use wording of this form only when appropriate:

> The thesis still contributes a transparent and leakage-aware evaluation
> workflow, but the empirical support for the intended decoding claim was
> weaker than expected under the locked confirmatory design.

## Prohibited wording

Do not use the result language below unless a future protocol explicitly broadens
scope and the evidence actually supports it.

- "The model reads emotions from the brain."
- "The result generalizes to people in general."
- "The coefficients identify the neural basis of emotion."
- "The system is clinically useful."
- "The result proves causation."
- "Transfer performance validates universal emotion representation."

## Adjudication workflow

Apply the rules in this order:

1. verify confirmatory validity;
2. classify the primary claim;
3. classify the transfer claim;
4. classify the interpretability claim;
5. classify the overall thesis contribution;
6. write the thesis text using the approved scope language.

Do not start at step 5.

## Maintenance rule

Update this file whenever any of the following changes:

- the locked confirmatory protocol changes;
- the primary metric changes;
- the required control policy changes;
- transfer is promoted from secondary to primary;
- interpretability is assigned a different evidential role;
- the thesis contribution statement changes.

## Status

Current status: suitable as a repository-level claim adjudication document for
pre-results thesis preparation.
