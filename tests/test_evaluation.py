"""Unit tests for evaluation: the test-set metrics, prediction, evaluate_run, MAC counting,
and the loss-curve figures showing once in a notebook."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from edgecnn.config.loader import load_config
from edgecnn.contracts import paths, schema
from edgecnn.contracts.schema import ContractViolation
from edgecnn.contracts.types import CHECKPOINT_KEYS

torch = pytest.importorskip("torch", reason="torch not installed")

from torch import nn  # noqa: E402

from edgecnn.evaluation.benchmark import count_macs  # noqa: E402
from edgecnn.evaluation.metrics import (  # noqa: E402
    _test_metrics_payload,
    compute_metrics,
    evaluate_run,
    predict,
)

NAMES = ["A", "B", "C"]
RUN = "model_b__adam__seed42"


def _experiment(name: str) -> str:
    return str(paths.CONFIGS_DIR / "experiments" / f"{name}.yaml")


# --- compute_metrics -------------------------------------------------------------


def test_metrics_match_a_hand_computed_case() -> None:
    #        true: 0  0  1  1  2  2
    #   predicted: 0  1  1  1  2  0
    result = compute_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], NAMES, RUN, "model_b")
    assert result.confusion_matrix == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert result.accuracy == pytest.approx(4 / 6)
    assert [result.per_class[n]["precision"] for n in NAMES] == pytest.approx([1 / 2, 2 / 3, 1])
    assert [result.per_class[n]["recall"] for n in NAMES] == pytest.approx([1 / 2, 1, 1 / 2])
    assert [result.per_class[n]["f1"] for n in NAMES] == pytest.approx([1 / 2, 4 / 5, 2 / 3])
    assert [result.per_class[n]["support"] for n in NAMES] == [2, 2, 2]
    assert result.macro_precision == pytest.approx((1 / 2 + 2 / 3 + 1) / 3)
    assert result.macro_recall == pytest.approx(2 / 3)


def test_rows_are_true_classes_and_columns_predictions() -> None:
    result = compute_metrics([0], [1], ["A", "B"], RUN, "model_b")
    assert result.confusion_matrix == [[0, 1], [0, 0]]


def test_a_class_never_predicted_scores_zero_and_the_matrix_stays_square() -> None:
    result = compute_metrics([0, 1, 2], [0, 1, 1], [*NAMES, "D"], RUN, "model_b")
    assert [len(row) for row in result.confusion_matrix] == [4, 4, 4, 4]
    assert result.per_class["C"]["precision"] == 0.0  # C is never predicted
    assert result.per_class["D"]["support"] == 0  # D is absent altogether


def test_results_hold_only_plain_python_numbers() -> None:
    result = compute_metrics(np.array([0, 1]), np.array([0, 0]), ["A", "B"], RUN, "model_b")
    json.dumps(dataclasses.asdict(result))  # NumPy scalars would fail here


def test_bad_inputs_are_refused() -> None:
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0], NAMES, RUN, "model_b")
    with pytest.raises(ValueError):
        compute_metrics([], [], NAMES, RUN, "model_b")
    with pytest.raises(ValueError):
        compute_metrics([0, 3], [0, 1], NAMES, RUN, "model_b")


def test_metrics_file_satisfies_its_contract() -> None:
    result = compute_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], NAMES, RUN, "model_b")
    payload = _test_metrics_payload(result, None)
    schema.validate(payload, "test_metrics")
    assert payload["num_samples"] == 6 and payload["split"] == "test"
    assert payload["weighted_recall"] == pytest.approx(result.accuracy)  # always equal


# --- predict ---------------------------------------------------------------------


class _Scores(nn.Module):
    """Returns its input as the class scores, so the prediction is the largest input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_predict_takes_the_top_score_in_loader_order_and_restores_the_mode() -> None:
    from torch.utils.data import DataLoader, TensorDataset

    scores = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.0, 0.2, 0.7], [0.3, 0.6, 0.1]])
    loader = DataLoader(TensorDataset(scores, torch.tensor([1, 0, 2, 0])), batch_size=3)
    model = _Scores().train()
    y_true, y_pred = predict(model, loader, "cpu")
    assert y_true.tolist() == [1, 0, 2, 0] and y_pred.tolist() == [1, 0, 2, 1]
    assert y_true.dtype == np.int64 and model.training


# --- count_macs ------------------------------------------------------------------


def test_macs_of_single_layers() -> None:
    standard = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    assert count_macs(standard, (3, 64, 64)) == 64 * 64 * 8 * 3 * 9 == 884_736
    depthwise = nn.Conv2d(8, 8, 3, padding=1, groups=8)
    assert count_macs(depthwise, (8, 32, 32)) == 32 * 32 * 8 * 9
    assert count_macs(nn.Sequential(nn.Flatten(), nn.Linear(12, 5)), (3, 2, 2)) == 60


def test_macs_reproduce_torchvisions_published_figures() -> None:
    from torchvision import models

    for name, weights in (
        ("mobilenet_v2", models.MobileNet_V2_Weights.IMAGENET1K_V1),
        ("squeezenet1_1", models.SqueezeNet1_1_Weights.IMAGENET1K_V1),
    ):
        macs = count_macs(getattr(models, name)(weights=None), (3, 224, 224))
        assert round(macs / 1e9, 3) == weights.meta["_ops"]


def test_counting_leaves_the_model_untouched() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4)).train()
    before = model[1].running_mean.clone()
    count_macs(model, (3, 16, 16))
    assert model.training and torch.equal(model[1].running_mean, before)
    assert not model[0]._forward_hooks


# --- evaluate_run ----------------------------------------------------------------


@pytest.fixture
def trained_model_b(monkeypatch, tmp_path):
    """Model B trained for one epoch on the synthetic data; checkpoints and metrics in tmp."""
    from edgecnn.data import build_dataloaders
    from edgecnn.models.registry import build_model_from_config
    from edgecnn.training.trainer import Trainer

    monkeypatch.setattr(paths, "DEBUG_CHECKPOINTS_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(paths, "METRICS_DIR", tmp_path / "metrics")
    cfg = load_config(_experiment("model_b__adam"), mode="synthetic", **{"training.epochs": 1})
    data = build_dataloaders(cfg)
    model = build_model_from_config(cfg, data.num_classes)
    trained = Trainer(cfg).fit(model, data, cfg)
    return cfg, data, trained, model


def test_evaluate_run_scores_the_whole_test_split(trained_model_b) -> None:
    cfg, data, trained, _ = trained_model_b
    result = evaluate_run(cfg, data, trained.checkpoint_path)
    assert sum(map(sum, result.confusion_matrix)) == len(data.test.dataset) == 30
    assert result.class_names == data.class_names and 0.0 <= result.accuracy <= 1.0
    assert not paths.test_metrics_json(cfg.run_id).exists()  # synthetic runs never write


def test_evaluate_run_scores_the_trained_weights(trained_model_b) -> None:
    cfg, data, trained, model = trained_model_b  # one epoch, so best.pt holds the final weights
    y_true, y_pred = predict(model, data.test, next(model.parameters()).device)
    expected = compute_metrics(y_true, y_pred, data.class_names, cfg.run_id, cfg.model_name)
    assert evaluate_run(cfg, data, trained.checkpoint_path) == expected


def test_evaluate_run_finds_the_best_checkpoint_itself(trained_model_b) -> None:
    cfg, data, trained, _ = trained_model_b
    found = evaluate_run(cfg, data)
    given = evaluate_run(cfg, data, trained.checkpoint_path)
    assert found.confusion_matrix == given.confusion_matrix


def test_a_checkpoint_from_another_run_is_refused(trained_model_b, tmp_path) -> None:
    cfg, data, trained, _ = trained_model_b
    blob = torch.load(trained.checkpoint_path, weights_only=False)
    blob["class_names"] = list(reversed(blob["class_names"]))
    wrong = tmp_path / "wrong.pt"
    torch.save(blob, wrong)
    with pytest.raises(ContractViolation, match="class_names"):
        evaluate_run(cfg, data, wrong)


def test_official_evaluation_writes_valid_test_metrics(fake_eurosat, monkeypatch, tmp_path) -> None:
    from edgecnn.data import build_dataloaders, prepare_dataset
    from edgecnn.models.registry import build_model_from_config

    monkeypatch.setattr(paths, "METRICS_DIR", tmp_path / "metrics")
    cfg = fake_eurosat.config("official")
    prepare_dataset(cfg)  # writes the fake split, as notebook 01 does for real
    data = build_dataloaders(cfg)
    model = build_model_from_config(cfg, data.num_classes)
    checkpoint = {
        "model_name": cfg.model_name,
        "state_dict": model.state_dict(),
        "epoch": 1,
        "val_acc": 0.0,
        "class_names": data.class_names,
        "input_shape": list(data.input_shape),
        "seed": cfg.seed,
        "config_snapshot": dict(cfg.raw),
    }
    assert set(CHECKPOINT_KEYS) <= set(checkpoint)
    torch.save(checkpoint, tmp_path / "best.pt")

    result = evaluate_run(cfg, data, tmp_path / "best.pt")
    written = schema.read_json(paths.test_metrics_json(cfg.run_id), "test_metrics")
    assert written["num_samples"] == 9 and written["accuracy"] == result.accuracy
    assert written["checkpoint"].endswith("best.pt")
    assert b"\r\n" not in paths.test_metrics_json(cfg.run_id).read_bytes()


# --- loss-curve figures show once in a notebook ----------------------------------


def test_curve_figures_come_back_closed_so_notebooks_show_them_once() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from edgecnn.evaluation.curves import plot_loss_curves, plot_optimizer_overlay

    epochs = [
        {"epoch": e, "train_loss": 1 / e, "val_loss": 1.2 / e, "train_acc": 0.1 * e,
         "val_acc": 0.09 * e, "lr": 0.001, "epoch_time_s": 1.0}
        for e in (1, 2, 3)
    ]
    history = {"run_id": RUN, "model": "model_b", "optimizer": "adam", "epochs": epochs}
    for fig in (plot_loss_curves(history), plot_optimizer_overlay([history])):
        assert not plt.fignum_exists(fig.number)
