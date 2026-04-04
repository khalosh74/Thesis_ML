from __future__ import annotations

from pathlib import Path

import pytest

from Thesis_ML.config.config_registry import load_config_registry
from Thesis_ML.config.runtime_selection import resolve_runtime_config_path


def test_runtime_selection_explicit_path_wins_over_alias(tmp_path: Path) -> None:
    explicit_path = tmp_path / "explicit_protocol.json"
    explicit_path.write_text("{}\n", encoding="utf-8")

    resolved = resolve_runtime_config_path(
        explicit_path,
        "protocol.thesis_confirmatory_frozen",
        default_alias="protocol.thesis_confirmatory_frozen",
        fallback_path=tmp_path / "fallback.json",
    )
    assert resolved == explicit_path.resolve()


def test_runtime_selection_resolves_explicit_alias() -> None:
    resolved = resolve_runtime_config_path(
        None,
        "protocol.thesis_canonical_default",
        default_alias="protocol.thesis_confirmatory_frozen",
        fallback_path=Path("configs/protocols/thesis_confirmatory_v1.json"),
    )
    assert resolved.name == "thesis_canonical_nested_v2.json"


def test_runtime_selection_resolves_default_alias_when_no_override() -> None:
    resolved = resolve_runtime_config_path(
        None,
        None,
        default_alias="comparison.grouped_nested_default",
        fallback_path=Path("configs/comparisons/model_family_grouped_nested_comparison_v2.json"),
    )
    assert resolved.name == "model_family_grouped_nested_comparison_v2.json"


def test_runtime_selection_resolves_explicit_thesis_runtime_registry_alias() -> None:
    resolved = resolve_runtime_config_path(
        None,
        "registry.decision_support_thesis_runtime",
        default_alias="registry.decision_support_thesis_runtime",
        fallback_path=Path("configs/decision_support_registry_revised_execution.json"),
    )
    assert resolved.name == "decision_support_registry_revised_execution.json"


def test_runtime_selection_resolves_explicit_package_registry_alias() -> None:
    resolved = resolve_runtime_config_path(
        None,
        "registry.decision_support_package_default",
        default_alias="registry.decision_support_thesis_runtime",
        fallback_path=Path("configs/decision_support_registry_revised_execution.json"),
    )
    assert resolved.name == "decision_support_registry.json"


def test_backup_registry_is_not_present_in_active_config_registry() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payload = load_config_registry(registry_path=repo_root / "configs" / "config_registry.json")
    for entry in payload.get("configs", []):
        path_text = str(entry.get("path", ""))
        assert "decision_support_registry_revised_execution.E02_backup.json" not in path_text


def test_runtime_selection_default_alias_uses_fallback_when_registry_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = tmp_path / "fallback_protocol.json"
    fallback.write_text("{}\n", encoding="utf-8")

    def _stub_resolve_alias(alias: str, **kwargs):
        if kwargs.get("fallback") is not None:
            return Path(kwargs["fallback"]).resolve()
        raise FileNotFoundError("Config registry file not found.")

    monkeypatch.setattr(
        "Thesis_ML.config.runtime_selection.resolve_config_alias", _stub_resolve_alias
    )
    resolved = resolve_runtime_config_path(
        None,
        None,
        default_alias="protocol.thesis_confirmatory_frozen",
        fallback_path=fallback,
    )
    assert resolved == fallback.resolve()


def test_runtime_selection_explicit_alias_does_not_silently_fallback_when_registry_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = tmp_path / "fallback_protocol.json"
    fallback.write_text("{}\n", encoding="utf-8")

    def _stub_resolve_alias(alias: str, **kwargs):
        if kwargs.get("fallback") is not None:
            return Path(kwargs["fallback"]).resolve()
        raise FileNotFoundError("Config registry file not found.")

    monkeypatch.setattr(
        "Thesis_ML.config.runtime_selection.resolve_config_alias", _stub_resolve_alias
    )
    with pytest.raises(FileNotFoundError):
        resolve_runtime_config_path(
            None,
            "protocol.thesis_confirmatory_frozen",
            default_alias="protocol.thesis_confirmatory_frozen",
            fallback_path=fallback,
        )


def test_runtime_selection_unknown_alias_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        resolve_runtime_config_path(
            None,
            "unknown.alias",
            default_alias="protocol.thesis_confirmatory_frozen",
            fallback_path=Path("configs/protocols/thesis_confirmatory_v1.json"),
        )
