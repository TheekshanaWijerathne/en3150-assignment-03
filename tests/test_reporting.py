"""Unit tests for notebook 07's tables and figures, built from fabricated results for all
six runs, so a broken table shows up before the official runs rather than after them."""

from __future__ import annotations

import warnings

import pytest

from edgecnn.contracts import paths, schema

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")

from edgecnn.evaluation import reporting  # noqa: E402

#: run_id: (learning rate, parameters, size in KB, MACs, test accuracy)
RUNS = {
    "model_b__sgd__seed42": (0.01, 71_818, 304.1, 12_989_568, 0.8975),
    "model_b__sgd_momentum__seed42": (0.01, 71_818, 304.1, 12_989_568, 0.9612),
    "model_b__adam__seed42": (0.001, 71_818, 304.1, 12_989_568, 0.961),
    "model_a__adam__seed42": (0.001, 1_143_466, 4_475.0, 42_337_536, 0.975),
    "mobilenet_v2__adam__seed42": (0.001, 2_236_682, 8_747.5, 6_312_896, 0.95),
    "squeezenet1_1__adam__seed42": (0.001, 727_626, 2_850.2, 21_900_000, 0.93),
}

#: Validation accuracy per epoch, by optimizer.
VAL_ACC = {
    "sgd": [0.5] * 5 + [0.85] * 10 + [0.9] * 5,  # best 0.90 at epoch 16; 90% of it from epoch 6
    "sgd_momentum": [0.7] * 2 + [0.95] * 18,  # best 0.95 at epoch 3; 90% of it from epoch 3
    "adam": [0.9] + [0.96] * 19,  # best 0.96 at epoch 2; 90% of it from epoch 1
}

#: Fine-tuning leaves part of each backbone frozen.
TRAINABLE = {"mobilenet_v2__adam__seed42": 1_218_890, "squeezenet1_1__adam__seed42": 391_306}

T4 = "cuda (Tesla T4)"


def _write_run(run_id: str) -> None:
    model, optimizer, _ = paths.parse_run_id(run_id)
    lr, params, size_kb, macs, accuracy = RUNS[run_id]
    curve = VAL_ACC[optimizer]
    best = curve.index(max(curve))
    epochs = [
        {
            "epoch": i + 1,
            "train_loss": 1.0,
            "val_loss": 1.0,
            "train_acc": acc,
            "val_acc": acc,
            "lr": lr,
            "epoch_time_s": 12.5,
        }
        for i, acc in enumerate(curve)
    ]
    history = {
        "run_id": run_id,
        "model": model,
        "optimizer": optimizer,
        "seed": 42,
        "device": T4,
        "epochs": epochs,
        "best_epoch": best + 1,
        "best_val_acc": curve[best],
        "hyperparameters": {"optimizer": optimizer, "lr": lr},
    }
    per_class = {name: {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 5} for name in "AB"}
    metrics = {
        "run_id": run_id,
        "model": model,
        "split": "test",
        "accuracy": accuracy,
        "macro_precision": accuracy - 0.01,
        "macro_recall": accuracy - 0.02,
        "macro_f1": accuracy - 0.015,
        "per_class": per_class,
        "confusion_matrix": [[5, 0], [0, 5]],
        "class_names": ["A", "B"],
        "num_samples": 10,
    }
    resources = {
        "run_id": run_id,
        "model": model,
        "trainable_params": TRAINABLE.get(run_id, params),
        "total_params": params,
        "model_size_kb": size_kb,
        "macs": macs,
        "mean_epoch_time_s": 12.5,
        "inference_latency_ms": 3.0,
        "peak_mem_mb": 1.25,
        "device": "cpu (test)",
    }
    schema.write_json(paths.history_json(run_id), history, "history")
    schema.write_json(paths.test_metrics_json(run_id), metrics, "test_metrics")
    schema.write_json(paths.resources_json(run_id), resources, "resources")


@pytest.fixture
def results(monkeypatch, tmp_path):
    """All six runs' committed JSON, in a temporary results/ beside a temporary report/."""
    tables = {
        name: getattr(paths, name).name
        for name in (
            "CUSTOM_COMPARISON_TABLE",
            "OPTIMIZER_COMPARISON_TABLE",
            "FINAL_COMPARISON_TABLE",
        )
    }
    monkeypatch.setattr(paths, "METRICS_DIR", tmp_path / "results" / "metrics")
    monkeypatch.setattr(paths, "FIGURES_DIR", tmp_path / "results" / "figures")
    monkeypatch.setattr(paths, "TABLES_DIR", tmp_path / "results" / "tables")
    monkeypatch.setattr(paths, "REPORT_FIGURES_DIR", tmp_path / "report" / "figures")
    for name, filename in tables.items():
        monkeypatch.setattr(paths, name, tmp_path / "results" / "tables" / filename)
    for run_id in RUNS:
        _write_run(run_id)
    return tmp_path


def test_every_run_with_test_metrics_is_found(results) -> None:
    assert sorted(reporting.load_all_runs()) == sorted(RUNS)


def test_section_4_table_gives_precision_and_recall_for_both_models(results) -> None:
    table = reporting.build_custom_comparison_table(write=False)
    assert "| Model A | 1,143,466 | 4,475.0 | 42,337,536 | 12.5 | 97.50 | 96.50 | 95.50 |" in table
    assert "| Model B | 71,818 | 304.1 | 12,989,568 | 12.5 | 96.10 | 95.10 | 94.10 |" in table


def test_section_3_table_reports_convergence_and_final_accuracy(results) -> None:
    table = reporting.build_optimizer_comparison_table(write=False)
    assert "| SGD | 0.01 | 16 | 90.00 | 89.75 | 6 | 12.5 |" in table
    assert "| SGD + momentum | 0.01 | 3 | 95.00 | 96.12 | 3 | 12.5 |" in table
    assert "| Adam | 0.001 | 2 | 96.00 | 96.10 | 1 | 12.5 |" in table


def test_section_6_table_gives_size_in_kb_and_mb(results) -> None:
    table = reporting.build_final_comparison_table(write=False)
    assert (
        "| SqueezeNet 1.1 | 391,306 | 727,626 | 2,850.2 | 2.78 | 21,900,000 | 3.00 | 1.25 "
        "| 93.00 | 92.00 | 91.00 |"
    ) in table
    assert "| MobileNetV2 | 1,218,890 | 2,236,682 |" in table
    assert "| Model B | 71,818 | 71,818 |" in table


def test_parameter_scatter_counts_every_stored_weight(results) -> None:
    axis = reporting.plot_tradeoff_scatter("total_params", write=False).axes[0]
    points = {text.get_text(): text.xy[0] for text in axis.texts}
    assert points == {"Model B": 71_818, "MobileNetV2": 2_236_682, "SqueezeNet 1.1": 727_626}
    assert axis.get_xscale() == "log"  # 71k to 2.2M: more than a decade
    assert axis.get_xlabel() == "Total parameters (log scale)"
    assert axis.get_legend() is None  # Model B is cheapest and most accurate: no frontier


def test_mac_scatter_draws_the_frontier_through_the_undominated_models(results) -> None:
    figure = reporting.plot_tradeoff_scatter("macs", write=False)
    axis = figure.axes[0]
    (frontier,) = axis.get_lines()
    assert list(frontier.get_xdata()) == [6_312_896, 12_989_568]  # MobileNetV2, then Model B
    assert [text.get_text() for text in axis.get_legend().get_texts()] == ["Pareto frontier"]
    # 6.3M to 21.9M spans less than a decade: a linear axis, labelled in millions
    assert axis.get_xscale() == "linear" and axis.get_xlabel() == "MACs per image"
    figure.canvas.draw()
    labels = [label.get_text() for label in axis.get_xticklabels() if label.get_text()]
    assert labels and all(label.endswith("M") for label in labels)


def test_nothing_is_written_unless_asked(results) -> None:
    reporting.build_custom_comparison_table(write=False)
    reporting.build_optimizer_comparison_table(write=False)
    reporting.build_final_comparison_table(write=False)
    reporting.plot_tradeoff_scatter("macs", write=False)
    assert reporting.export_report_figures(write=False) == []
    assert not (results / "results" / "tables").exists()
    assert not (results / "report").exists()


def test_official_tables_and_figures_land_where_the_report_reads_them(results) -> None:
    curves = paths.curves_png("model_b__adam__seed42")
    curves.parent.mkdir(parents=True)
    curves.write_bytes(b"png")  # a figure written earlier by notebook 03

    for build in (
        reporting.build_custom_comparison_table,
        reporting.build_optimizer_comparison_table,
        reporting.build_final_comparison_table,
    ):
        assert build(write=True).startswith("*")
    for table in (
        paths.CUSTOM_COMPARISON_TABLE,
        paths.OPTIMIZER_COMPARISON_TABLE,
        paths.FINAL_COMPARISON_TABLE,
    ):
        assert b"\r\n" not in table.read_bytes()

    figure = reporting.plot_tradeoff_scatter("macs", write=True)
    reporting.plot_tradeoff_scatter("total_params", write=True)
    axis = figure.axes[0]
    assert sorted(text.get_text() for text in axis.texts) == [
        "MobileNetV2",
        "Model B",
        "SqueezeNet 1.1",
    ]
    assert axis.get_title() == "Test accuracy versus MACs"

    exported = reporting.export_report_figures(write=True)
    report = results / "report" / "figures"
    assert sorted(path.relative_to(report).as_posix() for path in exported) == [
        "accuracy_vs_macs.png",
        "accuracy_vs_params.png",
        "model_b__adam__seed42/curves.png",
    ]
    assert all(path.exists() for path in exported)


def test_runs_trained_on_different_hardware_are_flagged(results) -> None:
    run = "model_a__adam__seed42"
    history = schema.read_json(paths.history_json(run), "history")
    history["device"] = "cpu (Intel Xeon CPU @ 2.20GHz)"
    schema.write_json(paths.history_json(run), history, "history")

    with pytest.warns(UserWarning, match="not trained on one device"):
        groups = reporting.check_same_hardware([run, "model_b__adam__seed42"])
    assert groups == {history["device"]: [run], T4: ["model_b__adam__seed42"]}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reporting.check_same_hardware(["model_b__sgd__seed42", "model_b__adam__seed42"])


def test_a_missing_result_names_its_run(results) -> None:
    paths.resources_json("model_a__adam__seed42").unlink()
    with pytest.raises(FileNotFoundError, match="model_a__adam__seed42"):
        reporting.build_custom_comparison_table(write=False)
