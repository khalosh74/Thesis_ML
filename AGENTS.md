# Repository instructions

This repository is a scientific ML experimentation framework. Treat confirmatory behavior as a hard scientific contract.

Rules:
1. Never weaken confirmatory safeguards for convenience.
2. Prefer minimal, high-confidence changes that fit existing architecture.
3. Before editing, inspect the relevant files and explain the implementation plan briefly.
4. After changes, run the narrowest relevant tests first, then broader checks if needed.
5. If dependencies prevent running tests, state exactly what blocked execution.
6. Do not rename public protocol/config files unless required.
7. Preserve backward compatibility for exploratory mode unless the task explicitly says otherwise.
8. For confirmatory features, enforce failures loudly rather than silently falling back.

For this repo, always check these areas when relevant:
- configs/protocols/
- schemas/
- docs/confirmatory/
- src/Thesis_ML/experiments/
- src/Thesis_ML/verification/
- tests/

When making confirmatory changes, ensure:
- schema validation exists
- protocol preflight validation exists
- target mapping hash is enforced
- deviations are logged
- interpretation limits appear in final reporting
- tests cover both pass and fail cases