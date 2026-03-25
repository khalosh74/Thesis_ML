# Method and AI Disclosure

## Purpose

This document records the software environment, analysis tools, coding tools,
writing-support tools, and AI-use verification rules relevant to this thesis
project. It is intended to support transparent method reporting in the thesis
method chapter and transparent tool disclosure in the discussion and reflection
materials.

The document is repository-facing rather than thesis-facing. It should be kept
accurate as the project evolves and used as the source document when drafting:

- the **Software tools and implementation** subsection of the method chapter;
- the **Use of AI tools** subsection required by the thesis structure;
- the **IT tools / AI tools** part of the discussion and reflection document.

## Scope and maintenance rule

This file should contain only information that is either:

1. verified directly from the repository, configuration files, or generated
   artifacts; or
2. explicitly confirmed by the thesis author.

Do not use this file to guess versions, workflows, or tool usage that have not
been verified.

Update this file whenever any of the following changes:

- the canonical runtime or dependency lockfile changes;
- the main analysis toolchain changes;
- a new AI tool is used for planning, coding, analysis, or writing;
- verification practice changes;
- the final thesis text adopts wording that no longer matches actual practice.

## Disclosure principles

The project follows these disclosure principles.

1. **Truthfulness over polish.** Only report tools that were actually used.
2. **Function over branding.** State what each tool was used for, not only its
   name.
3. **Human accountability.** AI outputs may support work, but scientific
   judgment, claim selection, interpretation, and final responsibility remain
   with the thesis author.
4. **No AI output as evidence by itself.** Scientific claims in the thesis must
   be supported by literature, reasoning, or project artifacts rather than by
   the fact that an AI system produced text or code.
5. **Verification before adoption.** No AI-generated text, code, claim,
   configuration, or citation should be included in the thesis or official
   repository outputs without explicit human review.

## Verified project software stack

The following items are directly supported by the current repository files.

### Core execution environment

| Tool / system | Status | Role in the project | Evidence basis |
|---|---|---|---|
| Python 3.13 | Verified canonical runtime | Main execution environment for data handling, feature construction, model execution, artifact generation, and tests | `.python-version`, `README.md`, `pyproject.toml`, `Dockerfile` |
| `uv` lockfile workflow | Verified canonical dependency path | Reproducible dependency installation and command execution | `README.md`, `uv.lock`, `Dockerfile`, bootstrap scripts |
| Docker / devcontainer | Verified optional development environment | Reproducible containerized development environment | `Dockerfile`, `.devcontainer/devcontainer.json` |
| Git repository | Verified version-control basis | Code versioning and provenance tracking | `.git/`, repository structure |

### Analysis and modelling tools

| Tool / package | Status | Role in the project | Notes |
|---|---|---|---|
| SPM first-level GLM outputs | Verified upstream analysis dependency | Produces first-level beta maps and associated metadata used as ML inputs | The repository consumes these outputs; first-level estimation is upstream of the Python repo |
| `numpy` | Verified dependency | Numerical array computation | Core scientific computing layer |
| `pandas` | Verified dependency | Tabular data handling, dataset indexing, report tables, CSV/JSON processing | Core data-management layer |
| `nibabel` | Verified dependency | Neuroimaging file handling for NIfTI data | Domain-specific file I/O |
| `scikit-learn` | Verified dependency | Supervised learning, evaluation, and baseline modelling | Core ML toolkit |
| `pydantic` | Verified dependency | Configuration and schema validation | Supports robust contract validation |
| `openpyxl` | Verified dependency | Workbook generation and spreadsheet handling | Used for experiment program and workbook workflows |
| `scipy` | Verified optional dependency | Optional scientific routines, including SPM-related support where needed | Optional extra in `pyproject.toml` |
| `optuna` | Verified optional dependency | Optional search mode for decision-support workflows | Optional extra in `pyproject.toml` |

### Quality, reproducibility, and engineering tools

| Tool / process | Status | Role in the project | Notes |
|---|---|---|---|
| `pytest` | Verified dev dependency | Automated tests | Used for regression and acceptance checks |
| `ruff` | Verified dev dependency | Linting and formatting policy support | Repository-configured quality tool |
| `mypy` | Verified dev dependency | Static type checking for covered modules | Improves implementation trustworthiness |
| `pre-commit` | Verified dev dependency | Local quality automation support | Present in development extras |
| Artifact verification scripts | Verified | Confirmatory readiness, official artifact completeness, reproducibility verification, bundle verification | `scripts/verify_*`, `scripts/acceptance_smoke.py` |
| Dataset fingerprinting / git provenance | Verified project policy | Reproducibility and auditability for official runs | Described in repository README and official artifact workflow |

## Current repository workflow map

The repository implements a staged workflow that should be disclosed in the
method chapter as part of tool usage and method application.

| Workflow stage | Main tools | What they are used for |
|---|---|---|
| Upstream first-level modelling | SPM | Creation of first-level GLM beta maps and associated regressors / masks |
| Dataset organization | Python, `pandas`, project CLI | Index creation, metadata handling, and dataset contract checks |
| Feature handling | Python, `nibabel`, project CLI | Feature extraction / caching from NIfTI-derived inputs |
| Model execution | Python, `scikit-learn`, project CLI | Baselines, model fitting, evaluation, and comparison/protocol execution |
| Reporting artifacts | Python, JSON / CSV writers, `openpyxl` | Structured reports, workbook outputs, machine-readable artifacts |
| Reproducibility checks | `uv`, `pytest`, verification scripts | Locked dependency installation, tests, official-run verification |

## Suggested method-chapter disclosure of software tools

The following points should be covered in the thesis method chapter.

1. **Upstream analysis context.** First-level GLM estimation was performed in
   SPM, producing condition-level beta maps and associated metadata that serve
   as inputs to the ML pipeline.
2. **Python implementation.** Downstream data handling, model execution,
   reporting, and validation were implemented in Python.
3. **Core packages.** The main analysis stack includes `numpy`, `pandas`,
   `nibabel`, and `scikit-learn`, with `pydantic` and repository-specific
   schemas used for configuration validation and `openpyxl` used for workbook
   generation.
4. **Reproducibility path.** Dependency installation and canonical execution
   follow the `uv` lockfile workflow, and the repository includes tests and
   verification scripts for official outputs.
5. **Scope note.** The repository is a local research framework for
   reproducible experimentation, not a clinical deployment system.

## AI-tool disclosure policy

AI tools must be disclosed transparently whenever they are used for the thesis
process. The disclosure should name the tool, state what it was used for, and
state how the outputs were verified before adoption.

### Allowed AI-support categories

AI-support uses may be disclosed under one or more of the following categories.

| Category | Typical use in this project | Disclosure expectation |
|---|---|---|
| Planning support | Structuring experiments, drafting implementation plans, organizing next steps | State that AI supported planning but did not determine final scientific claims on its own |
| Coding support | Reviewing code, suggesting refactors, drafting helper code, generating documentation | State how code suggestions were manually reviewed and tested |
| Writing support | Improving structure, wording, concision, or consistency of thesis prose or repo documentation | State that the author reviewed, edited, and took responsibility for all final text |
| Analysis support | Helping structure result presentation, identify checks, or critique argumentation | State that AI output was not accepted as evidence without checking artifacts and literature |
| Administrative support | Checklists, traceability documents, disclosure templates, or reflection scaffolding | State that these were support materials rather than direct scientific evidence |

### Uses that must not be represented misleadingly

The following uses should **not** be described as if the AI tool performed the
scientific reasoning independently:

- selecting final research claims without human review;
- serving as a source for scientific facts without literature verification;
- generating references that enter the thesis without bibliographic checking;
- generating code that enters official runs without inspection and testing;
- interpreting final results without comparison to actual run artifacts.

## Current known AI-use status

Based on the current thesis draft and repository work, the following AI-use
statements are already supportable.

| Usage area | Current status | Confidence |
|---|---|---|
| Planning support | Supported by current project work and repo-documentation workflow | High |
| Revision / text-structure support | Supported by current thesis draft wording about planning, revision support, and text development | High |
| Documentation drafting | Supported by current repository-documentation work | High |
| Coding assistance | Possible / likely, but should only be reported if specifically confirmed in the AI log | Medium |
| Analysis interpretation support | Possible / likely, but should only be reported if specifically confirmed in the AI log | Medium |

Before final thesis submission, convert this section from a status summary into
an exact record based on `docs/ai_usage_log.csv`.

## Mandatory verification rules for AI outputs

Every AI-assisted output that is adopted into the thesis or repository should be
verified according to the category below.

### A. Verification of AI-assisted prose

Before AI-assisted prose is adopted:

1. confirm that the text matches the actual thesis scope and does not introduce
   stronger claims than the evidence supports;
2. remove unsupported factual statements or replace them with verified sources;
3. check terminology against the thesis glossary and locked definitions;
4. revise the wording so the final phrasing is genuinely owned by the author;
5. ensure that thesis-specific claims are supported by literature or project
   artifacts rather than by the AI output itself.

### B. Verification of AI-assisted code or configuration

Before AI-assisted code or configuration is adopted:

1. inspect the logic line by line;
2. confirm alignment with the protocol, comparison spec, or workflow contract;
3. run relevant tests or smoke checks where possible;
4. confirm that paths, schema keys, metrics, and split logic match repository
   policy;
5. review whether the change could introduce leakage, broken provenance,
   unstable defaults, or invalid comparisons.

### C. Verification of AI-assisted scientific claims or citations

Before AI-assisted claims or citations are adopted:

1. verify bibliographic details against the bibliography manager or source file;
2. confirm that the cited source actually supports the sentence being written;
3. remove fabricated, weak, or irrelevant references;
4. distinguish between background context, methodological justification, and
   empirical evidence;
5. do not let AI-generated wording obscure uncertainty or dataset limitations.

### D. Verification of AI-assisted interpretation

Before AI-assisted interpretation is adopted:

1. check the actual run artifacts, metrics, figures, and logs;
2. confirm that the interpretation follows the locked claim-evaluation rules;
3. confirm that within-person and cross-person evidence remain separated;
4. keep all scope limitations attached to the claim;
5. reject any interpretation that implies causality, clinical readiness,
   localization certainty, or population generalization without supporting
   evidence.

## Minimum log requirement

Every meaningful AI-assisted interaction that influenced the thesis or
repository should be logged in `docs/ai_usage_log.csv`.

At minimum, each log entry should record:

- date;
- thesis phase or deliverable;
- tool name and model/version if known;
- task category;
- short description of the prompt/input;
- short description of the output;
- whether the output was adopted, partially adopted, or rejected;
- how the output was verified;
- what human edits were made;
- which final artifact, chapter, or repo file was affected.

## Recommended wording for the thesis

The wording below is a **starting point** and must be edited to match actual
use before submission.

### Method chapter: software tools and implementation

> First-level GLM estimation was performed in SPM, producing condition-level
> beta maps and associated metadata used as inputs to the machine-learning
> pipeline. Downstream data handling, feature processing, model execution, and
> structured reporting were implemented in Python 3.13 using a local research
> framework. Core analysis libraries included NumPy, pandas, nibabel, and
> scikit-learn, with pydantic used for configuration validation and openpyxl
> used for workbook-based experiment support. Reproducible execution was managed
> through a lockfile-based `uv` workflow together with automated tests and
> verification scripts for official outputs.

### Method chapter or dedicated AI-tools subsection

> AI-based tools were used selectively for planning support, documentation and
> prose revision, and, where applicable, coding assistance. AI outputs were not
> treated as scientific evidence. Any adopted text, code, or claims were
> manually reviewed, edited, and verified against repository artifacts,
> methodological constraints, and scientific sources before inclusion in the
> thesis or project outputs.

### Discussion / reflection material

> AI tools were useful mainly for structuring work, identifying gaps, improving
> wording, and drafting support documents. Their outputs still required careful
> verification because they could overstate claims, omit scope limitations, or
> suggest text that sounded plausible without being sufficiently grounded in the
> thesis evidence.

## Final pre-submission checklist

Before thesis submission, confirm the following:

- the disclosed software tools match the actual final workflow;
- every reported AI tool has at least one corresponding row in the AI usage log;
- the final thesis text uses the same scope limits as the repo governance
  documents;
- no AI-generated citation entered the final bibliography without verification;
- no AI-generated interpretation entered the final thesis without being checked
  against actual results;
- the reflection document examples are consistent with the AI log.
