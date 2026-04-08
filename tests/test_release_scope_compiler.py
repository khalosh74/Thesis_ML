from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tests._release_test_utils import dataset_manifest_path, make_temp_release_bundle, repo_root
from Thesis_ML.release.loader import load_dataset_manifest, load_release_bundle
from Thesis_ML.release.scope import compile_release_scope


def _write_manifest_with_extra_audiovisual_task(tmp_path: Path) -> Path:
    source_manifest_path = dataset_manifest_path()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_index_path = (source_manifest_path.parent / source_manifest["index_csv"]).resolve()
    source_index = pd.read_csv(source_index_path)
    modified_index = source_index.copy()
    modified_index.loc[0, "task"] = "other_audiovisual_task"
    modified_index.loc[0, "modality"] = "audiovisual"

    modified_index_path = tmp_path / "dataset_index_with_extra_task.csv"
    modified_index.to_csv(modified_index_path, index=False)

    manifest = dict(source_manifest)
    manifest["index_csv"] = str(modified_index_path.resolve())
    manifest["data_root"] = str((repo_root() / "demo_data" / "synthetic_v1" / "data_root").resolve())
    manifest["cache_dir"] = str((repo_root() / "demo_data" / "synthetic_v1" / "cache").resolve())
    manifest_path = tmp_path / "dataset_manifest_with_extra_task.json"
    manifest_path.write_text(f"{json.dumps(manifest, indent=2)}\n", encoding="utf-8")
    return manifest_path


def test_release_scope_compiler_excludes_non_scope_audiovisual_tasks(tmp_path: Path) -> None:
    release_path = make_temp_release_bundle(tmp_path)
    manifest_path = _write_manifest_with_extra_audiovisual_task(tmp_path)
    release = load_release_bundle(release_path)
    dataset = load_dataset_manifest(manifest_path)

    result = compile_release_scope(
        release_bundle=release,
        dataset_manifest=dataset,
        run_dir=tmp_path / "run",
    )

    selected_tasks = sorted(result.selected_index_df["task"].astype(str).unique().tolist())
    assert selected_tasks == ["emo", "recog"]
    assert "other_audiovisual_task" not in set(selected_tasks)

    selected_modalities = sorted(result.selected_index_df["modality"].astype(str).unique().tolist())
    assert selected_modalities == ["audiovisual"]

    scope_manifest_payload = json.loads(result.scope_manifest_path.read_text(encoding="utf-8"))
    assert int(scope_manifest_payload["selected_row_count"]) == int(len(result.selected_index_df))
    assert len(str(scope_manifest_payload["selected_sample_ids_sha256"])) == 64
    assert (
        int(
            scope_manifest_payload["exclusions_summary"]["by_reason"][
                "task_not_in_release_scope"
            ]
        )
        >= 1
    )
