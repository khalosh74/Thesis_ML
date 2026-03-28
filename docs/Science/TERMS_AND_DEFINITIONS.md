# Terms and Definitions

## Purpose

This document defines the central scientific and repository terms used in this
project. Its purpose is to keep the thesis, protocols, run artifacts, and
reporting language consistent.

The definitions in this file are thesis-facing and execution-facing at the same time. Each term is defined once, linked to its operational meaning in the
repository, and paired with usage boundaries where confusion is likely.

This file supports the thesis requirement that key concepts be easy to find and
used consistently. It also implements the research standard requirement that
important terms such as stability, generalization, transfer, interpretability,
and the emotion label be defined once and then reused without drift.

## Authority order

When terminology conflicts across documents, use this precedence order:

1. Locked confirmatory protocol and protocol-bound artifacts.
2. Thesis research question, contribution statement, and scope limits.
3. This definitions file.
4. Other repository documentation.

This means the present file is a harmonization document, not a license to
change the scientific scope after the fact.

## Usage rules

- Use one preferred term for each concept in thesis-facing writing.
- Avoid upgrading a narrow term into a broader one. For example, do not write
  *generalization* when the evidence only supports *transfer* or *internal
  validation*.
- Keep scientific meaning and operational meaning aligned. If a term refers to
  a locked split, target, or metric, name the exact object when possible.
- If a future protocol changes a science-critical definition, update this file
  and the thesis-facing documents together.

## Definitions table

| Term | Preferred definition in this project | How it is operationalized here | Prefer / avoid | Notes and scope limits |
|---|---|---|---|---|
| **Problem** | The general-interest scientific problem addressed by this work is that apparent success in repeated-session fMRI decoding can be misleading when leakage, hierarchical dependence, confounds, and over-interpretation are not controlled. | Stated in thesis framing and implemented through leakage-aware split logic, controls, governance rules, and scoped interpretation. | Prefer: `problem`, `scientific problem`. Avoid: vague labels such as `challenge` when the thesis-level problem is meant. | This is one methodological problem with closely related sub-parts, not multiple unrelated problems. |
| **Research question** | The main research question asks whether repeated-session fMRI beta-map representations support reliable within-person decoding of a three-class coarse-affect target under strict leakage-aware held-out-session evaluation, and to what extent the learned mapping transfers across individuals under domain shift. | Reflected in the thesis umbrella RQ, RQ1-RQ3, and the confirmatory/secondary protocol structure. | Prefer: `research question`, `main research question`, `RQ1`, `RQ2`, `RQ3`. Avoid: introducing new untracked research questions in repo docs. | There should be one main RQ and a small number of tightly related sub-questions. |
| **Contribution statement** | The project contributes an empirically evaluated, leakage-aware modelling and validation framework for repeated-session fMRI decoding that separates within-person cross-session generalization from cross-person transfer and treats interpretability as supporting robustness evidence. | Encoded in thesis framing, traceability rules, claim evaluation rules, and confirmatory governance. | Prefer: `contribution statement`. Avoid: inconsistent rephrasings that change novelty or scope. | The novelty is methodological and evidence-strategy based, not primarily architectural. |
| **Computer and systems sciences contribution** | The thesis contributes to computer and systems sciences through trustworthy machine learning for high-dimensional repeated-measures scientific data. | Expressed in thesis framing and supported by repository controls for evaluation, reproducibility, and interpretation. | Prefer this exact field positioning. Avoid: reframing the work as only neuroscience or only software engineering. | The work sits at the intersection of ML evaluation methodology, reproducible experimentation, and interpretation-aware analysis. |
| **Repeated-session fMRI** | A dataset setting in which the same participant is measured over multiple scanning sessions, making observations hierarchically dependent rather than simply independent. | Data hierarchy is subject -> session -> run/BAS -> event-level beta sample. | Prefer: `repeated-session`, `repeated-measures`. Avoid: language implying IID samples. | This dependency structure is the reason split logic and leakage control are science-critical. |
| **Dense within-person sampling** | Repeatedly measuring the same individual across many sessions to estimate stable and variable components of that person’s responses more directly. | Conceptual justification for treating within-person cross-session evaluation as the primary setting. | Prefer: `dense within-person sampling` when discussing the conceptual basis. Avoid: treating it as equivalent to population sampling. | This term explains why within-person stability and cross-person transfer are separate scientific questions. |
| **Sample unit** | The basic modeled observation in this project. | Locked as `beta_event` in the confirmatory plan. | Prefer: `event-level beta sample` or `beta_event`. Avoid: generic `trial` or `scan` unless that is what is actually meant. | Use the exact sample-unit name when describing data flow or metrics. |
| **Beta map** | A voxelwise first-level GLM parameter-estimate map used as the feature-bearing representation for one modeled event or condition. | Derived upstream from SPM first-level GLM estimation and then masked/processed in the ML pipeline. | Prefer: `beta map`, `event-level beta map`. Avoid: calling it a raw fMRI time series. | The downstream ML acts on GLM-derived maps, not raw BOLD sequences. |
| **Target** | The label the model is asked to predict. | The locked primary target is `coarse_affect`. | Prefer: `target`, `target label`. Avoid: switching between target and outcome terms without need. | The target is an operationalization, not the full psychological phenomenon. |
| **`coarse_affect`** | The locked three-class target used in the confirmatory study, consisting of `positive`, `neutral`, and `negative`. | Defined by `affect_mapping_v2` loaded from the locked target asset `configs/targets/affect_mapping_v2.json` in the confirmatory plan. | Prefer the exact code-facing name when referring to the locked target. Avoid: `emotion` when the mapped target is actually meant. | This target is a coarse operationalization and should not be described as exhaustive emotion truth. |
| **Label operationalization** | The explicit mapping from source conditions to the target classes used for modeling. | Implemented by the locked mapping version `affect_mapping_v2` loaded from the versioned target asset. | Prefer: `label operationalization`, `target mapping`. Avoid: implying the classes are natural kinds discovered by the model. | Successful decoding is evidence under the chosen formulation only. |
| **Within-person cross-session generalization** | The ability of a model trained on some sessions from one participant to predict the target on held-out sessions from that same participant. | Primary scientific setting; implemented by `within_subject_loso_session`. | Prefer: `within-person cross-session generalization`. Avoid: shortening this to only `generalization` when ambiguity exists. | This is the primary form of evidence for stable predictive signal in this thesis. |
| **Held-out-session evaluation** | An evaluation design in which full sessions are withheld from training and used only for testing. | In the locked primary split, each subject is trained on all but one session and tested on the held-out session. | Prefer: `held-out-session evaluation`, `session-held-out evaluation`. Avoid: `cross-validation` alone when the grouping unit matters. | The held-out session is the critical anti-leakage boundary for the primary claim. |
| **`within_subject_loso_session`** | The locked primary split contract for within-person analysis. LOSO here means leave-one-session-out within each subject. | For each subject independently: train on all but one session, test on the held-out session, with no session overlap. | Prefer the exact split name in protocol-facing writing. Avoid: ambiguous references such as `LOSO` without saying what is left out. | This is the primary confirmatory split, not merely one exploratory option. |
| **Cross-person transfer** | The extent to which a model trained on one participant can be applied to the other participant. | Secondary analysis only; reported separately from within-person performance. | Prefer: `cross-person transfer`. Avoid: calling this `generalization` without qualification. | In this thesis, transfer is more demanding and more weakly justified than within-person evidence. |
| **Frozen cross-person transfer** | Cross-person transfer in which the trained model is applied to the target participant without re-training, retuning, or target-person adaptation. | Implemented by the secondary split `frozen_cross_person_transfer`. | Prefer the full phrase when protocol fidelity matters. Avoid: any wording that suggests fine-tuning on the target person. | The frozen requirement is essential to the scope of the transfer claim. |
| **Domain shift** | A change in the data-generating conditions between training and test domains that makes transfer harder and scientifically distinct from within-person testing. | Here, the shift is primarily between individuals, with all the associated person-specific structure. | Prefer: `domain shift` when explaining why cross-person transfer is difficult. Avoid: treating transfer failure as equivalent to no within-person signal. | Cross-person weakness does not invalidate within-person evidence. |
| **Internal validation** | Evaluation performed within the declared dataset and scope of the study, using locked train-test separation and controls. | The confirmatory thesis evidence is internal to the locked dataset and protocol. | Prefer: `internal validation`. Avoid: `external validation` unless an independent dataset is actually used. | Internal validation supports narrower claims than external validation. |
| **External validation** | Evaluation on a separate dataset or setting not used to define the primary confirmatory evidence. | Not part of the current primary confirmatory claim. | Prefer: `external validation` only for truly separate data/settings. Avoid: presenting secondary compatibility checks as equivalent to the main confirmatory evidence. | External validation may be valuable later but is not required to define the current thesis framing. |
| **Leakage** | Any flow of information from test data or test structure into training, preprocessing, tuning, or model selection that makes performance estimates too optimistic or scientifically invalid. | Primary risk in repeated-session data; addressed through session-level splitting, in-pipeline preprocessing, controls, and audit artifacts such as `leakage_audit.json`. | Prefer: `leakage` or `data leakage`. Avoid: reducing it to only obvious row duplication. | Leakage can arise through split design, preprocessing, feature selection, tuning, or repeated-subject dependence. |
| **Leakage-aware evaluation** | An evaluation design that treats leakage prevention as a scientific requirement rather than a software convenience. | Operationalized by locked split contracts, train-only fitting of preprocessing, confirmatory governance, and required controls. | Prefer: `leakage-aware evaluation`. Avoid: implying ordinary CV is automatically sufficient. | This is a defining feature of the contribution. |
| **Confound** | Non-target structure that can influence predictions and create performance that is not scientifically attributable to the intended signal. | In this setting, examples include session structure, motion-related artifacts, modality/task structure, or other nuisance patterns. | Prefer: `confound`, `non-target structure`. Avoid: using `noise` when the structure is systematic. | A strong score can still be scientifically weak if it is confound-driven. |
| **Primary claim** | The main confirmatory claim predeclared for the thesis. | In the locked plan: a fixed linear classifier predicts `coarse_affect` above chance under `within_subject_loso_session`, evaluated by `balanced_accuracy`. | Prefer: `primary claim`. Avoid: reassigning primary status post hoc to another result. | All secondary or exploratory results remain subordinate to this claim. |
| **Primary metric** | The one metric declared in advance as decisive for the main claim. | Locked as `balanced_accuracy`. | Prefer: `primary metric`. Avoid: choosing the best-looking metric after results. | Secondary metrics may contextualize but cannot replace the primary metric. |
| **Balanced accuracy** | The mean of per-class recall values, used to reduce distortion from class imbalance in multiclass classification. | Locked as the confirmatory `primary_metric`. | Prefer: `balanced_accuracy` in protocol-facing contexts and `balanced accuracy` in prose. Avoid: calling it plain `accuracy`. | This metric is used because a class-imbalanced setting can make ordinary accuracy misleading. |
| **Secondary metrics** | Additional metrics used to contextualize, not replace, the primary metric. | In the locked plan: `macro_f1`, confusion matrix, and class-wise recall. | Prefer: `secondary metrics`. Avoid: promoting them over the primary metric after execution. | Use them to interpret error structure, not to override the decision rule. |
| **Dummy baseline** | A simple reference classifier run under the same split structure to show what trivial performance looks like. | Required control analysis for confirmatory validity. | Prefer: `dummy baseline`. Avoid: comparing primary results only to informal expectations. | Exceeding the dummy baseline is necessary but not sufficient for support. |
| **Label permutation test** | A control in which training labels are shuffled within the locked evaluation structure to estimate a chance-reference null distribution. | Required with `1000` permutations in the locked confirmatory plan. | Prefer: `label permutation test`, `permutation control`. Avoid: vague phrases such as `random baseline` when the actual null procedure matters. | The primary result must survive the permutation-based threshold under the locked protocol. |
| **Confirmatory** | A mode of analysis governed by a locked protocol, fixed science-critical choices, required controls, and auditable artifacts suitable for primary thesis evidence. | In repo mode metadata: `framework_mode=confirmatory`, `canonical_run=true`, with protocol-bound execution and readiness checks. | Prefer: `confirmatory`. Avoid: using it for ad hoc runs. | Confirmatory status can be downgraded if a science-critical deviation occurs. |
| **Exploratory** | Analysis used for development, diagnostics, or idea generation without claim-deciding authority for the primary thesis claim. | In repo mode metadata: `framework_mode=exploratory`, `canonical_run=false`. | Prefer: `exploratory`. Avoid: citing exploratory runs as if they were confirmatory support. | Exploratory analysis is valuable, but it cannot rescue a failed primary confirmatory result. |
| **Canonical run** | A run designated as an official protocol-governed evidence artifact rather than an ad hoc experiment. | In confirmatory mode, `canonical_run=true`. | Prefer: `canonical run` when referring to official evidence outputs. Avoid: calling every run official. | This term is about evidence status, not model quality by itself. |
| **Interpretability** | Analysis intended to characterize which features most influence model predictions and whether those patterns are stable enough to support cautious understanding of model behavior. | Implemented through locked interpretability procedures and summaries such as `interpretability_summary.json`. | Prefer: `interpretability`, `model-behavior interpretation`. Avoid: equating it with direct brain localization. | Interpretability here provides supporting evidence, not definitive mechanistic explanation. |
| **Interpretability-driven robustness evidence** | Evidence from explanation patterns that is used to test whether model behavior shows recurring, plausible structure across folds or sessions. | Evaluated as a separate claim layer in `CLAIM_EVALUATION_RULES.md`. | Prefer this full phrase in thesis-facing discussion. Avoid: `the explanation proves the biology`. | This evidence is supportive and bounded, not primary proof of neural mechanism. |
| **Model coefficients / weights** | Parameters of the fitted discriminative model that help determine predictions. | Produced by linear classifiers in the locked design. | Prefer: `model coefficients`, `weights`. Avoid: calling them activation maps or anatomical truth. | Linear weights are not automatically interpretable as neural activation patterns. |
| **Robustness** | The extent to which an observed result or explanation pattern remains meaningful under conservative validation and relevant checks. | Here it includes stability across held-out folds/sessions, control analyses, and governance compliance. | Prefer: `robustness` when tied to specific checks. Avoid: using it as a vague synonym for high accuracy. | Robustness is multi-part: performance, controls, stability, and protocol fidelity. |
| **Stability** | Recurrent or consistent predictive or explanatory structure across held-out sessions, folds, or repeated measurements. | Operationalized through session-held-out results and interpretability summaries across folds/sessions. | Prefer: `stability`. Avoid: treating one good split as evidence of stability. | Stability is central to RQ1 and RQ3 and must be evaluated, not assumed. |
| **Reproducibility** | The ability to rerun the declared workflow and recover the documented analysis structure and outputs under the same conditions. | Supported by config snapshots, environment information, manifests, fingerprints, and standardized outputs. | Prefer: `reproducibility`. Avoid: confusing it with external generalization. | A result can be reproducible yet still scientifically narrow. |
| **Auditability** | The ability to trace a thesis-facing claim back to the exact protocol, configuration, data contract, and artifact set that produced it. | Supported by manifests, protocol IDs, artifact paths, and traceability documentation. | Prefer: `auditability`. Avoid: vague claims of transparency without traceable evidence. | Auditability is part of the methodological contribution. |
| **Subgroup analysis** | A split or report restricted to a predefined subset such as subject, task, or modality. | Allowed as descriptive analysis subject to minimum subgroup thresholds in the confirmatory plan. | Prefer: `subgroup analysis`. Avoid: treating subgroup findings as replacements for failed primary evidence. | In the locked plan, subgroup analyses are descriptive unless explicitly promoted. |
| **Scope limit / non-claim** | An explicit boundary on what the study does not justify even if the main result is positive. | Includes no clinical-use claim, no causal claim, no population-level generalization claim, and no localization claim from coefficients alone. | Prefer: `scope limit`, `non-claim`. Avoid: leaving these boundaries implicit. | These are part of the scientific framing, not after-the-fact disclaimers. |
| **Cross-case evidence** | Evidence about transfer between the two available participants in this dataset, without claiming population-level generalization. | Used to describe cross-person findings under the current `N = 2` repeated-session setting. | Prefer: `cross-case evidence`. Avoid: `population evidence` for the current transfer results. | This wording is required for conservative interpretation. |

## Synonym and wording guardrails

Use the following wording rules to avoid drift.

| Preferred wording | Use when | Avoid |
|---|---|---|
| `within-person cross-session generalization` | Referring to the primary scientific setting | `generalization` alone when cross-person transfer could also be meant |
| `cross-person transfer` | Referring to frozen application from one participant to the other | `cross-person generalization` if it suggests broad external validity |
| `held-out-session evaluation` | Referring to the primary split logic | `cross-validation` alone |
| `coarse_affect` | Referring to the locked target object | `emotion` when the mapped three-class label is actually meant |
| `balanced accuracy` | Referring to the main confirmatory metric in prose | `accuracy` |
| `interpretability-driven robustness evidence` | Referring to the role of explanation analyses in this thesis | `proof of localization`, `brain mechanism evidence` |
| `internal validation` | Referring to confirmatory evidence in the current dataset | `external validation` or `real-world validation` |
| `cross-case evidence` | Referring to transfer findings with two participants | `population generalization` |
| `predictive association` | Referring to what positive decoding can support | `causal effect`, `causal mechanism` |

## Minimum usage requirements in thesis-facing reporting

When a result is reported from this repository, the write-up should, at a
minimum, state all of the following explicitly:

- the target (`coarse_affect` if using the locked confirmatory target);
- the split or evaluation setting (`within_subject_loso_session` or the exact
  secondary split name);
- whether the evidence is confirmatory, secondary, or exploratory;
- the primary metric (`balanced_accuracy` for the locked confirmatory claim);
- the relevant scope limit if the wording could otherwise overclaim.

A report is considered terminologically incomplete if it presents a score
without making those elements clear.

## Relationship to other repository documents

This file is intended to be read together with:

- [`SCIENTIFIC_POSITIONING.md`](./SCIENTIFIC_POSITIONING.md)
- [`THESIS_TRACEABILITY_MATRIX.md`](./THESIS_TRACEABILITY_MATRIX.md)
- [`CLAIM_EVALUATION_RULES.md`](./CLAIM_EVALUATION_RULES.md)
- [`DATASET_SCOPE_AND_GENERALIZATION.md`](./DATASET_SCOPE_AND_GENERALIZATION.md)
- [`confirmatory/confirmatory_plan_v1.md`](./confirmatory/confirmatory_plan_v1.md)

If terminology in those files changes in a science-relevant way, update this
file in the same change set.

## Maintenance rule

Update this document whenever any of the following changes:

- the umbrella research question or any thesis sub-question changes;
- the contribution statement changes;
- the locked target definition changes;
- the primary split or transfer split changes;
- the primary metric changes;
- the role of interpretability changes;
- the scope limits or non-claims change.

Do not update wording merely to make results sound stronger after execution.
If the evidence is narrow, mixed, or negative, the terminology should preserve
that fact rather than hide it.
