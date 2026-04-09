# AGENT 06 — INTERPRETABILITY / STABILITY

## Branch
`agent/interpretability/stability`

## Mission
Build the interpretability family as supporting evidence about model behavior, with explicit emphasis on stability and interpretation limits.

## Scientific Role
This agent operationalizes the thesis position that interpretability is not direct localization.

Its job is to produce explanation artifacts that are:
- clearly secondary/supporting
- stability-aware
- bounded in interpretation
- useful for auditing model behavior

## In Scope
- explanation generation workflows approved by the contract
- fold stability summaries
- rerun stability summaries
- representative pattern visualizations
- guardrails that distinguish predictive coefficients from activation-like interpretations
- family-specific reports/tests

## Out of Scope
- causal claims
- direct localization claims from raw discriminative weights
- redefining the main confirmatory evidence
- transfer or robustness logic except by approved interface

## Owned Areas
Suggested ownership:
- interpretability bundle/config definitions
- explanation/stability reports
- interpretability family tests
- interpretability-specific visualization generation owned by this family

## Read-Only Dependencies
- confirmatory semantics
- shared execution/report substrate
- contract claim boundaries

## Required Outputs
1. Official interpretability bundles.
2. Explanation stability summaries.
3. Representative pattern views with attached caveats.
4. Metadata/report text that prevents overinterpretation.

## Required Tests
- tests that interpretability artifacts are labeled as supporting evidence
- tests that raw coefficients are not presented through misleading semantics in official outputs
- tests for stability summary generation integrity

## Merge Criteria
- interpretability outputs are scientifically bounded
- stability is quantified rather than implied
- outputs cannot be mistaken for direct localization evidence

## Review Checklist
- Does this family help audit behavior rather than decorate results?
- Are explanation caveats encoded, not merely implied?
- Is stability treated as a measurable property?

## Notes for the Agent
The value of this family lies in disciplined interpretation, not visual flourish.
