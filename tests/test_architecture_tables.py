"""Unit tests for the Section 2 architecture tables."""

from __future__ import annotations

import pytest

from edgecnn.config.loader import load_config
from edgecnn.contracts import paths

torch = pytest.importorskip("torch", reason="torch not installed")

from torch import nn  # noqa: E402

from edgecnn.models.custom.summary import architecture_tables, layer_table  # noqa: E402
from edgecnn.models.registry import build_model_from_config  # noqa: E402


def _small_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU6(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False),
        nn.Conv2d(8, 16, 1, bias=False),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10),
    )


def _model(name: str) -> nn.Module:
    cfg = load_config(paths.CONFIGS_DIR / "experiments" / f"{name}__adam.yaml", mode="synthetic")
    return build_model_from_config(cfg, 10)


def test_every_layer_shows_its_formula_and_output_shape() -> None:
    table = layer_table(_small_model(), (3, 8, 8)).set_index("layer")
    assert table.loc["0", "formula"] == "3×3×3×8 = 216"
    assert table.loc["1", "formula"] == "2×8 = 16"  # batch-norm scale and shift
    assert table.loc["4", "type"] == "Conv2d depthwise"
    assert table.loc["4", "formula"] == "3×3×1×8 = 72"
    assert table.loc["5", "type"] == "Conv2d pointwise"
    assert table.loc["5", "formula"] == "1×1×8×16 = 128"
    assert table.loc["8", "formula"] == "16×10 + 10 = 170"
    assert table.loc["0", "output"] == "8×8×8" and table.loc["3", "output"] == "8×4×4"
    assert table.loc["6", "type"] == "Global average pool" and table.loc["8", "output"] == "10"
    assert table.loc["total", "params"] == 216 + 16 + 72 + 128 + 170


@pytest.mark.parametrize("name", ["model_a", "model_b"])
def test_totals_equal_the_trainable_parameters(name: str) -> None:
    model = _model(name).train()
    table = layer_table(model, (3, 64, 64))
    assert table["params"].iloc[-1] == sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert model.training  # measured on a copy


def test_tables_are_written_only_when_asked(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(paths, "CUSTOM_ARCHITECTURES_TABLE", tmp_path / "custom_architectures.md")
    monkeypatch.setattr(paths, "PARAM_BREAKDOWN_TABLE", tmp_path / "param_breakdown.md")
    model_a, model_b = _model("model_a"), _model("model_b")

    tables = architecture_tables(model_a, model_b, (3, 64, 64))
    assert not any(tmp_path.iterdir())
    trainable_b = sum(p.numel() for p in model_b.parameters() if p.requires_grad)
    assert "## Model B - main layers" in tables["custom_architectures"]
    assert f"{trainable_b:,}" in tables["custom_architectures"]
    assert "3×3×3×16" in tables["param_breakdown"]

    architecture_tables(model_a, model_b, (3, 64, 64), write=True)
    for name in ("custom_architectures", "param_breakdown"):
        assert (tmp_path / f"{name}.md").read_text(encoding="utf-8") == tables[name]
