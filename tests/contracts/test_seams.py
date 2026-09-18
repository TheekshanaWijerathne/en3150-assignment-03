"""Seam 1 and Seam 2 behaviour, once the stubs behind them are implemented.

Every test here uses the ``pending`` helper from conftest, so it SKIPS while
its subject still raises ``NotImplementedError`` and becomes a real assertion
the moment that stub is filled in. Nobody has to remember to enable them.

These need torch, so they are skipped entirely if torch is not installed.
"""

from __future__ import annotations

import pytest

from edgecnn.contracts.types import INPUT_SHAPE, MODEL_B_PARAM_BUDGET
from tests.conftest import pending

torch = pytest.importorskip("torch", reason="torch not installed")

pytestmark = pytest.mark.contract


# --- SEAM 2: the model registry --------------------------------------------


def test_registry_exposes_every_required_key() -> None:
    """All four models the assignment needs must be registered."""
    from edgecnn.models.registry import REQUIRED_KEYS, available_models

    with pending("Members 2 and 4"):
        available = set(available_models())
        missing = [key for key in REQUIRED_KEYS if key not in available]
        assert not missing, f"unregistered models: {missing}"


@pytest.mark.parametrize(
    ("name", "owner"),
    [
        ("model_a", "Member 2"),
        ("model_b", "Member 2"),
        ("mobilenet_v2", "Member 4"),
        ("squeezenet1_1", "Member 4"),
    ],
)
def test_model_returns_logits_of_the_right_shape(name: str, owner: str) -> None:
    """The Seam 2 forward contract, checked identically for all four models."""
    from edgecnn.models.registry import build_model

    num_classes = 10
    with pending(owner):
        model = build_model(name, num_classes=num_classes, input_shape=INPUT_SHAPE)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, *INPUT_SHAPE))

    assert out.shape == (2, num_classes), f"{name}: expected (2, {num_classes}), got {out.shape}"
    assert out.dtype == torch.float32
    assert getattr(model, "num_classes", None) == num_classes, f"{name}: must expose .num_classes"


@pytest.mark.parametrize("name", ["model_a", "model_b", "mobilenet_v2", "squeezenet1_1"])
def test_model_outputs_are_raw_logits_not_probabilities(name: str) -> None:
    """A row summing to 1.0 means a softmax was left inside the model.

    That trains against a double-softmaxed signal: it does not crash, it just
    quietly underperforms, which is why it needs a test rather than a review.
    """
    from edgecnn.models.registry import build_model

    with pending("Members 2 and 4"):
        model = build_model(name, num_classes=10, input_shape=INPUT_SHAPE)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, *INPUT_SHAPE))

    row_sums = out.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3), (
        f"{name}: output rows sum to 1 - remove the softmax, CrossEntropyLoss applies it"
    )


@pytest.mark.parametrize("name", ["model_a", "model_b", "mobilenet_v2", "squeezenet1_1"])
def test_model_works_at_batch_size_one(name: str) -> None:
    """Member 1 measures inference latency at batch size 1, as an edge device would."""
    from edgecnn.models.registry import build_model

    with pending("Members 2 and 4"):
        model = build_model(name, num_classes=10, input_shape=INPUT_SHAPE)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, *INPUT_SHAPE))

    assert out.shape == (1, 10)


# --- SEAM 1: the data bundle -----------------------------------------------


def test_synthetic_fixture_satisfies_the_data_contract(tmp_path) -> None:
    """The Phase 0 unblocker must itself satisfy Seam 1, or it is not a fixture."""
    from edgecnn.contracts.schema import validate_manifest
    from edgecnn.data.synthetic import make_synthetic_fixture

    with pending("Member 1"):
        root = make_synthetic_fixture(tmp_path, num_classes=4, images_per_class=10)

    manifest = root / "split_manifest.csv"
    assert manifest.exists(), "fixture must write a real split_manifest.csv"
    assert validate_manifest(manifest) == 40


def test_dataloaders_honour_the_batch_contract() -> None:
    """images float32 (B,3,64,64) normalised; labels int64 in [0, num_classes)."""
    from edgecnn.config.loader import load_config
    from edgecnn.contracts import paths
    from edgecnn.data import build_dataloaders

    config_path = paths.CONFIGS_DIR / "experiments" / "model_b__adam.yaml"
    with pending("Member 1"):
        cfg = load_config(config_path, **{"inputs.dataset.name": "synthetic"})
        data = build_dataloaders(cfg)
        images, labels = next(iter(data.train))

    assert images.dtype == torch.float32
    assert images.shape[1:] == INPUT_SHAPE
    assert labels.dtype == torch.int64
    assert int(labels.min()) >= 0
    assert int(labels.max()) < data.num_classes
    assert len(data.class_names) == data.num_classes


def test_splits_are_disjoint() -> None:
    """A sample in both train and test would invalidate every reported number."""
    import pandas as pd

    from edgecnn.contracts import paths

    if not paths.SPLIT_MANIFEST.exists():
        pytest.skip("split not generated yet - run scripts/prepare_data.py")

    frame = pd.read_csv(paths.SPLIT_MANIFEST)
    groups = {name: set(g["relative_path"]) for name, g in frame.groupby("split")}
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["val"].isdisjoint(groups["test"])


def test_split_proportions_match_the_assignment() -> None:
    """70 / 15 / 15, within rounding."""
    import pandas as pd

    from edgecnn.contracts import paths
    from edgecnn.contracts.types import SPLIT_FRACTIONS

    if not paths.SPLIT_MANIFEST.exists():
        pytest.skip("split not generated yet - run scripts/prepare_data.py")

    frame = pd.read_csv(paths.SPLIT_MANIFEST)
    total = len(frame)
    for split, expected in SPLIT_FRACTIONS.items():
        actual = (frame["split"] == split).sum() / total
        assert abs(actual - expected) < 0.01, f"{split}: {actual:.3f} vs expected {expected}"
