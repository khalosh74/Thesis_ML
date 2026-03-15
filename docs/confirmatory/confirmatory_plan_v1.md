# Confirmatory Analysis Plan
## Protocol ID
`thesis_confirmatory_v1`

## Status
Locked confirmatory plan

## Version
- Plan version: `v1.0`
- Protocol version: `v1.0`
- Target mapping version: `affect_mapping_v1`
- Split implementation version: `split_contract_v1`

---

# 1. Study identity

## 1.1 Title
Confirmatory evaluation of event-level fMRI coarse-affect classification from beta maps

## 1.2 Research question
Can event-level fMRI beta maps predict `coarse_affect` above chance under a locked within-subject held-out-session evaluation?

## 1.3 Primary claim
A fixed linear classifier trained on event-level fMRI beta maps achieves above-chance `balanced_accuracy` for `coarse_affect` under `within_subject_loso_session`.

## 1.4 Scope of claim
This study supports only:
- predictive association, not causation
- internal evaluation under the locked dataset and protocol
- the declared target and split only

This study does **not** support:
- clinical use
- causal interpretation
- population-level external generalization
- neural localization claims from model coefficients

---

# 2. Target definition

## 2.1 Primary target
- Target name: `coarse_affect`
- Source column: `emotion`
- Mapping version: `affect_mapping_v1`
- Allowed classes:
  - `positive`
  - `neutral`
  - `negative`

## 2.2 Mapping table
| Emotion | Coarse affect |
|---|---|
| joy | positive |
| amusement | positive |
| calm | neutral |
| sadness | negative |
| fear | negative |
| anger | negative |

Replace this table with the exact locked mapping used in code.

## 2.3 Unknown and invalid labels
- Unknown labels are excluded before modeling.
- Missing target values are excluded before modeling.
- No relabeling is allowed during confirmatory execution.

## 2.4 Unit of analysis
- Sample unit: `beta_event`

---

# 3. Dataset contract

## 3.1 Dataset source
Describe the exact dataset build used for confirmatory analysis.

## 3.2 Inclusion rules
- Must have valid `beta_path`
- Must have valid `mask_path`
- Must have valid target label after mapping
- Must belong to an eligible subject/session pair
- Must satisfy spatial compatibility checks

## 3.3 Exclusion rules
List all exclusion rules here.

## 3.4 Data hierarchy
- Subject
- Session
- BAS / run
- Event-level beta sample

## 3.5 Leakage statement
The primary leakage risk is contamination across repeated samples within subject/session structure. The locked split is designed to hold out full sessions within subject.

## 3.6 Dataset fingerprint
Record the dataset fingerprint used in the final run:
- Dataset fingerprint: `TO_BE_FILLED_FROM_RUN`

---

# 4. Primary analysis

## 4.1 Primary split
- Split name: `within_subject_loso_session`

## 4.2 Split contract
For each subject independently:
- train on all but one session
- test on the held-out session
- no overlap between train and test sessions
- no sample may appear in more than one test fold

## 4.3 Primary model
- Model family: `ridge`
- Hyperparameter policy: `fixed`
- Class-weight policy: `none`
- Random seed: `42`

## 4.4 Preprocessing
- Feature extraction from beta maps using the locked feature pipeline
- Standardization inside sklearn pipeline only
- No preprocessing step may be fitted on test data

## 4.5 Primary metric
- Primary metric: `balanced_accuracy`

## 4.6 Secondary metrics
- `macro_f1`
- confusion matrix
- class-wise recall

These are contextual only and do not replace the primary metric.

---

# 5. Control analyses

## 5.1 Required controls
The confirmatory run is invalid unless all of the following are present:
- dummy baseline
- label permutation test
- fold-level predictions
- fold-level split log

## 5.2 Dummy baseline
A dummy classifier must be run under the same split structure.

## 5.3 Permutation test
- Required: yes
- Number of permutations: `1000`
- Null procedure: train-label shuffling within the locked training/evaluation structure

## 5.4 Success condition
The primary result is only considered supportive if:
1. it exceeds the dummy baseline, and
2. it survives the permutation-based threshold under the locked protocol

---

# 6. Secondary analyses

## 6.1 Allowed secondary analyses
### A. Frozen cross-person transfer
- Split: `frozen_cross_person_transfer`
- Status: secondary
- Interpretation: descriptive support only unless separately corrected and justified

## 6.2 Prohibited secondary drift
The following are not allowed to change during confirmatory execution:
- target definition
- split logic
- metric policy
- model family
- class-weight policy
- subgroup thresholds

---

# 7. Subgroup policy

## 7.1 Allowed subgroup axes
- `subject`
- `task`
- `modality`

## 7.2 Minimum subgroup requirements
- Minimum samples per subgroup: `20`
- Minimum class diversity: at least `2` classes present
- Subgroups below threshold are marked `insufficient_data` and not interpreted

## 7.3 Role of subgroup analyses
Subgroup analyses are descriptive unless explicitly promoted to confirmatory in this plan.

## 7.4 Prohibited subgroup use
Subgroup findings may not be used to rescue a failed primary analysis.

---

# 8. Multiplicity policy

## 8.1 Primary hypothesis count
- Number of primary hypotheses: `1`

## 8.2 Primary threshold
- Alpha: `0.05`

## 8.3 Secondary policy
Choose one and lock it:
- descriptive only
- FDR corrected

Locked choice for this plan:
- `descriptive_only`

## 8.4 Exploratory analyses
Exploratory analyses must be labeled exploratory and cannot support the primary claim.

---

# 9. Interpretation limits

The final confirmatory report must include all of the following statements:

1. Results indicate predictive association, not causation.
2. Internal validation does not establish external generalization.
3. Linear coefficients are not evidence of anatomical localization.
4. Outputs are research results, not clinical decision support.
5. Secondary analyses are not primary evidence.
6. Any subgroup result below the locked threshold is not interpretable.

---

# 10. Reporting contract

The final report must contain:
1. protocol ID
2. dataset fingerprint
3. target mapping version and hash
4. split definition
5. model definition
6. primary metric
7. controls
8. primary result
9. secondary results
10. subgroup results
11. multiplicity statement
12. interpretation limits
13. deviations from protocol

---

# 11. Deviations policy

Any deviation from this plan must be logged explicitly.

## 11.1 Allowed deviation handling
- deviation must be recorded
- confirmatory status must be downgraded if deviation affects science-critical settings

## 11.2 Science-critical deviations
Any change to the following invalidates confirmatory status:
- target
- split
- primary metric
- model family
- hyperparameter policy
- class-weight policy
- control requirements
- subgroup thresholds
- multiplicity policy

---

# 12. Sign-off

## 12.1 Locked by
- Name:
- Date:

## 12.2 Final confirmation
I confirm that the final confirmatory analysis will be executed only under the locked conditions described in this document.