"""The seam formats are real, self-consistent, and reject bad data.

These tests run fully today - they need no model, no dataset and no torch.
If one of them fails after a ``git pull``, an interface changed.
"""

from __future__ import annotations

import copy
import json

import pytest

from edgecnn.contracts import paths
from edgecnn.contracts.schema import (
    SCHEMA_FILES,
    ContractViolation,
    load_schema,
    validate,
)

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("name", sorted(SCHEMA_FILES))
def test_schema_document_exists_and_parses(name: str) -> None:
    schema = load_schema(name)
    assert schema["$schema"].startswith("https://json-schema.org/")
    assert "description" in schema, f"{name}: every seam schema must document itself"


@pytest.mark.parametrize("name", sorted(SCHEMA_FILES))
def test_schema_is_valid_json_schema(name: str) -> None:
    """A malformed schema would silently validate everything."""
    import jsonschema

    jsonschema.Draft202012Validator.check_schema(load_schema(name))


def test_schema_files_all_present_on_disk() -> None:
    missing = [f for f in SCHEMA_FILES.values() if not (paths.SCHEMAS_DIR / f).exists()]
    assert not missing, f"missing schema documents: {missing}"


# --- history.json ----------------------------------------------------------


def test_valid_history_passes(valid_history: dict) -> None:
    validate(valid_history, "history")


def test_history_rejects_fewer_than_20_epochs(valid_history: dict) -> None:
    """The assignment mandates at least 20 epochs; a debug run must not commit."""
    payload = copy.deepcopy(valid_history)
    payload["epochs"] = payload["epochs"][:5]
    with pytest.raises(ContractViolation):
        validate(payload, "history")


def test_history_rejects_malformed_run_id(valid_history: dict) -> None:
    payload = copy.deepcopy(valid_history)
    payload["run_id"] = "ModelB-Adam-42"
    with pytest.raises(ContractViolation):
        validate(payload, "history")


def test_history_rejects_unknown_optimizer(valid_history: dict) -> None:
    payload = copy.deepcopy(valid_history)
    payload["optimizer"] = "superadam"
    with pytest.raises(ContractViolation):
        validate(payload, "history")


def test_history_rejects_accuracy_above_one(valid_history: dict) -> None:
    """Catches a percentage being written where a fraction was expected."""
    payload = copy.deepcopy(valid_history)
    payload["epochs"][0]["val_acc"] = 85.0
    with pytest.raises(ContractViolation):
        validate(payload, "history")


# --- test_metrics.json -----------------------------------------------------


def test_valid_test_metrics_passes(valid_test_metrics: dict) -> None:
    validate(valid_test_metrics, "test_metrics")


def test_test_metrics_split_is_locked_to_test(valid_test_metrics: dict) -> None:
    """Headline numbers must come from the held-out split."""
    payload = copy.deepcopy(valid_test_metrics)
    payload["split"] = "val"
    with pytest.raises(ContractViolation):
        validate(payload, "test_metrics")


def test_test_metrics_requires_confusion_matrix(valid_test_metrics: dict) -> None:
    payload = copy.deepcopy(valid_test_metrics)
    del payload["confusion_matrix"]
    with pytest.raises(ContractViolation):
        validate(payload, "test_metrics")


def test_confusion_matrix_is_consistent_with_num_samples(valid_test_metrics: dict) -> None:
    """Not enforceable in JSON Schema, so assert it here.

    A mismatch means the wrong split was evaluated.
    """
    total = sum(sum(row) for row in valid_test_metrics["confusion_matrix"])
    assert total == valid_test_metrics["num_samples"]


def test_confusion_matrix_is_square_and_matches_class_names(valid_test_metrics: dict) -> None:
    cm = valid_test_metrics["confusion_matrix"]
    n = len(valid_test_metrics["class_names"])
    assert len(cm) == n, "confusion matrix row count must equal the class count"
    assert all(len(row) == n for row in cm), "confusion matrix must be square"


# --- resources.json --------------------------------------------------------


def test_valid_resources_passes(valid_resources: dict) -> None:
    validate(valid_resources, "resources")


def test_resources_requires_device_string(valid_resources: dict) -> None:
    """Timings without hardware are meaningless, and Section 4 requires it."""
    payload = copy.deepcopy(valid_resources)
    del payload["device"]
    with pytest.raises(ContractViolation):
        validate(payload, "resources")


def test_resources_rejects_zero_params(valid_resources: dict) -> None:
    payload = copy.deepcopy(valid_resources)
    payload["trainable_params"] = 0
    with pytest.raises(ContractViolation):
        validate(payload, "resources")


# --- norm_stats.json -------------------------------------------------------


def test_norm_stats_locks_fitted_on_to_train() -> None:
    """Fitting normalisation on val or test leaks held-out information.

    The schema pins the field so the mistake cannot be committed silently.
    """
    good = {"mean": [0.3, 0.3, 0.3], "std": [0.1, 0.1, 0.1],
            "fitted_on": "train", "num_images": 100}
    validate(good, "norm_stats")

    leaky = dict(good, fitted_on="all")
    with pytest.raises(ContractViolation):
        validate(leaky, "norm_stats")


def test_norm_stats_rejects_zero_std() -> None:
    """A zero std would divide by zero at normalisation time."""
    bad = {"mean": [0.3], "std": [0.0], "fitted_on": "train", "num_images": 10}
    with pytest.raises(ContractViolation):
        validate(bad, "norm_stats")


# --- split_meta.json -------------------------------------------------------


def test_split_meta_requires_manifest_hash() -> None:
    """The hash is how anyone detects a silently regenerated split."""
    meta = {
        "dataset_name": "eurosat",
        "num_classes": 2,
        "class_names": ["A", "B"],
        "image_size": [64, 64],
        "seed": 42,
        "stratified": True,
        "fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
        "counts": {"train": 70, "val": 15, "test": 15, "total": 100},
        "manifest_sha256": "a" * 64,
        "created_utc": "2026-09-18T00:00:00Z",
    }
    validate(meta, "split_meta")

    bad = dict(meta, manifest_sha256="not-a-hash")
    with pytest.raises(ContractViolation):
        validate(bad, "split_meta")


def test_split_meta_rejects_resolution_above_64() -> None:
    """The assignment caps image resolution at 64x64."""
    meta = {
        "dataset_name": "x", "num_classes": 2, "class_names": ["A", "B"],
        "image_size": [128, 128], "seed": 42, "stratified": True,
        "fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
        "counts": {"train": 70, "val": 15, "test": 15, "total": 100},
        "manifest_sha256": "a" * 64, "created_utc": "2026-09-18T00:00:00Z",
    }
    with pytest.raises(ContractViolation):
        validate(meta, "split_meta")


# --- split_manifest rows ---------------------------------------------------


def test_manifest_row_rejects_backslash_path() -> None:
    """A Windows path makes the committed manifest unusable on Linux."""
    with pytest.raises(ContractViolation):
        validate(
            {"relative_path": "images\\a.jpg", "label_index": 0,
             "label_name": "A", "split": "train"},
            "split_manifest",
        )


def test_manifest_row_rejects_unknown_split() -> None:
    with pytest.raises(ContractViolation):
        validate(
            {"relative_path": "images/a.jpg", "label_index": 0,
             "label_name": "A", "split": "holdout"},
            "split_manifest",
        )


def test_committed_split_artifacts_are_valid_if_present() -> None:
    """Once Member 1 commits the split, these three must stay contract-valid.

    Skips until the data stage has been run and its outputs committed.
    """
    if not paths.SPLIT_META.exists():
        pytest.skip("split not generated yet - run scripts/prepare_data.py")

    from edgecnn.contracts.schema import validate_manifest

    meta = json.loads(paths.SPLIT_META.read_text(encoding="utf-8"))
    validate(meta, "split_meta", source=paths.SPLIT_META)

    rows = validate_manifest(paths.SPLIT_MANIFEST)
    assert rows == meta["counts"]["total"], (
        "manifest row count disagrees with split_meta counts.total - "
        "the split was regenerated without updating the metadata"
    )
