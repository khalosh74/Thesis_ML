Logging configuration for Thesis_ML

This project uses a centralized logging initializer at `src/Thesis_ML/logging_config.py`.

Environment variables

- `THESIS_ML_LOG_LEVEL` — log level (e.g. `DEBUG`, `INFO`, numeric levels allowed). Default: `INFO`.
- `THESIS_ML_LOG_FORMAT` — set to `json` to enable structured JSON logging (recommended for CI/centralized logs); any other value selects plain-text ISO timestamps.
- `THESIS_ML_LOG_FILE` — optional path to write a rotating log file (10 MB, 5 backups).
- `THESIS_ML_LOG_FORCE` — if truthy (`1`, `true`, `yes`) will force reconfiguration even if logging was already configured by another library.

Quick examples

Enable JSON logs locally (bash):

```bash
export THESIS_ML_LOG_FORMAT=json
python -m uv run thesisml-run-protocol --protocol configs/protocols/thesis_canonical_nested_v2.json --dry-run
```

Enable JSON logs in CI (GitHub Actions):

```yaml
jobs:
  build_verify:
    env:
      THESIS_ML_LOG_FORMAT: json
```

Notes

- Call `configure_logging()` early in CLI entrypoints (this repository already does so in main CLIs).
- Prefer JSON for CI and long-running experiment runs so logs can be parsed centrally.
