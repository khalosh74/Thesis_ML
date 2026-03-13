# Decision-Support Compatibility Note

This file is retained for compatibility/reference only.

Canonical decision-support/operator documentation:
- `docs/DECISION_SUPPORT_AUTOMATION.md`
- `docs/RUNBOOK.md`
- `docs/OPERATOR_GUIDE.md`

Canonical command:

```bash
thesisml-run-decision-support \
  --registry configs/decision_support_registry.json \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --output-root outputs/artifacts/decision_support \
  --all
```

Compatibility wrapper (deprecated but still supported):

```bash
python run_decision_support_experiments.py --all
```

Use canonical CLI examples for all new operator/developer documentation.
