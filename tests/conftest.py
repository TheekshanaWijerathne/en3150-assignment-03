"""Shared pytest fixtures and the not-yet-implemented helper.

The suite is designed to be GREEN from day one, on a repo full of stubs.

That is deliberate. If ``pytest tests/contracts`` were red while the stubs were
unimplemented, nobody would look at it, and the one signal that tells a member
"someone else's merge broke your interface" would be lost in the noise. So a
test whose subject raises ``NotImplementedError`` reports as *skipped*, and
turns into a real assertion the moment that stub is filled in - no test edit
required.

    pytest tests/contracts -q      # after every git pull
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from edgecnn.contracts import paths


@contextlib.contextmanager
def pending(owner: str) -> Iterator[None]:
    """Skip the enclosing test while its subject is still a stub.

    Usage::

        with pending("Member 2"):
            model = build_model("model_b", num_classes=10)
        assert count_params(model) <= 100_000

    Once ``build_model`` stops raising ``NotImplementedError``, the assertion
    below the block starts running for real.
    """
    try:
        yield
    except NotImplementedError as exc:
        pytest.skip(f"not implemented yet ({owner}): {exc}")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return paths.REPO_ROOT


@pytest.fixture(scope="session")
def experiment_configs() -> list[Path]:
    """Every composed experiment config."""
    return sorted((paths.CONFIGS_DIR / "experiments").glob("*.yaml"))


@pytest.fixture(scope="session")
def stage_configs() -> list[Path]:
    """Every stage config."""
    return sorted((paths.CONFIGS_DIR / "stages").glob("*.yaml"))


@pytest.fixture
def valid_history() -> dict:
    """A minimal history.json payload that satisfies the contract.

    Twenty epochs, because the schema enforces the assignment's 20-epoch
    minimum. Tests mutate a copy of this to check that violations are caught.
    """
    return {
        "run_id": "model_b__adam__seed42",
        "model": "model_b",
        "optimizer": "adam",
        "seed": 42,
        "device": "cpu (test)",
        "epochs": [
            {
                "epoch": i,
                "train_loss": 2.0 - i * 0.05,
                "val_loss": 2.1 - i * 0.05,
                "train_acc": 0.1 + i * 0.03,
                "val_acc": 0.1 + i * 0.028,
                "lr": 0.001,
                "epoch_time_s": 12.5,
            }
            for i in range(1, 21)
        ],
        "best_epoch": 20,
        "best_val_acc": 0.66,
        "total_train_time_s": 250.0,
    }


@pytest.fixture
def valid_test_metrics() -> dict:
    """A minimal test_metrics.json payload that satisfies the contract."""
    return {
        "run_id": "model_b__adam__seed42",
        "model": "model_b",
        "split": "test",
        "accuracy": 0.81,
        "macro_precision": 0.80,
        "macro_recall": 0.79,
        "macro_f1": 0.795,
        "per_class": {
            "ClassA": {"precision": 0.8, "recall": 0.78, "f1": 0.79, "support": 50},
            "ClassB": {"precision": 0.8, "recall": 0.80, "f1": 0.80, "support": 50},
        },
        "confusion_matrix": [[39, 11], [10, 40]],
        "class_names": ["ClassA", "ClassB"],
        "num_samples": 100,
    }


@pytest.fixture
def valid_resources() -> dict:
    """A minimal resources.json payload that satisfies the contract."""
    return {
        "run_id": "model_b__adam__seed42",
        "model": "model_b",
        "trainable_params": 94_312,
        "total_params": 94_312,
        "model_size_kb": 378.5,
        "macs": 12_400_000,
        "input_shape": [3, 64, 64],
        "mean_epoch_time_s": 12.5,
        "inference_latency_ms": 3.2,
        "peak_mem_mb": 18.4,
        "device": "cpu (test)",
    }
