"""Unit tests for the shared utilities: seeding, devices, logging, Markdown tables, plot style."""

from __future__ import annotations

import logging
import random

import numpy as np
import pytest

from edgecnn.utils.io import format_cell, markdown_table, write_markdown_table
from edgecnn.utils.logging import setup_logging

torch = pytest.importorskip("torch", reason="torch not installed")

from edgecnn.utils.device import describe_device, resolve_device  # noqa: E402
from edgecnn.utils.seed import seed_everything, worker_init_fn  # noqa: E402

# --- seeding -------------------------------------------------------------------


def test_seed_everything_makes_every_generator_repeat() -> None:
    seed_everything(7)
    first = (random.random(), np.random.rand(), torch.rand(3))
    seed_everything(7)
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_seed_everything_switches_deterministic_mode() -> None:
    seed_everything(0, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    seed_everything(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_worker_init_fn_derives_numpy_seed_from_torch() -> None:
    torch.manual_seed(123)
    worker_init_fn(0)
    assert np.random.rand() == np.random.RandomState(123).rand()


# --- devices -------------------------------------------------------------------


def test_resolve_device_cpu_and_auto() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_resolve_device_rejects_unknown_names() -> None:
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_resolve_device_fails_loudly_for_a_missing_gpu() -> None:
    with pytest.raises(RuntimeError):
        resolve_device("cuda:99")  # no CUDA at all, or no 100th GPU


def test_describe_device_names_the_cpu_without_trademark_marks() -> None:
    text = describe_device("cpu")
    assert text.startswith("cpu")
    assert "(R)" not in text and "(TM)" not in text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU on this machine")
def test_describe_device_names_the_gpu() -> None:
    assert describe_device(resolve_device("cuda")).startswith("cuda:0 (")


# --- logging -------------------------------------------------------------------


def test_setup_logging_does_not_stack_handlers() -> None:
    setup_logging()
    setup_logging()
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_edgecnn_handler", False)]
    assert len(ours) == 1


# --- Markdown tables -----------------------------------------------------------


def test_markdown_table_formats_and_right_aligns_numbers() -> None:
    text = markdown_table(
        ["model", "params", "accuracy"],
        [["model_b", 94312, 0.8123], ["model_a", 1234567, 0.85]],
        caption="Demo",
    )
    lines = text.splitlines()
    assert lines[0] == "*Demo*"
    assert "| model | params | accuracy |" in lines
    assert "|---|---:|---:|" in lines
    assert "| model_b | 94,312 | 0.8123 |" in lines


def test_markdown_table_rejects_ragged_rows() -> None:
    with pytest.raises(ValueError):
        markdown_table(["a", "b"], [[1]])


def test_markdown_table_keeps_a_column_format_in_every_row() -> None:
    text = markdown_table(["model", "accuracy"], [["A", 90.0], ["B", 96.1]], formats=[None, ".2f"])
    assert "| A | 90.00 |" in text and "| B | 96.10 |" in text
    assert "|---|---:|" in text  # still right-aligned
    with pytest.raises(ValueError):
        markdown_table(["a", "b"], [[1, 2]], formats=[None])


def test_format_cell_handles_pipes_missing_values_and_large_floats() -> None:
    assert format_cell("a|b") == "a\\|b"
    assert format_cell(None) == "—"
    assert format_cell(1234.56) == "1,234.6"
    assert format_cell(True) == "True"


def test_write_markdown_table_creates_parent_folders(tmp_path) -> None:
    path = write_markdown_table(tmp_path / "tables" / "demo.md", ["a"], [[1]])
    assert path.read_text(encoding="utf-8").startswith("| a |")


# --- plot style ----------------------------------------------------------------

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")

from edgecnn.evaluation.plotting import (  # noqa: E402
    FIGURE_DPI,
    INLINE_DPI,
    MODEL_COLORS,
    NEUTRAL_GREY,
    OPTIMIZER_COLORS,
    SPLIT_COLORS,
    annotate_hardware,
    apply_style,
    color_for,
    label_for,
    save_figure,
)


def test_apply_style_sets_inline_and_saved_resolutions() -> None:
    import matplotlib.pyplot as plt

    apply_style()
    assert plt.rcParams["figure.dpi"] == INLINE_DPI
    assert plt.rcParams["savefig.dpi"] == FIGURE_DPI


def test_color_for_known_unknown_and_invalid_kinds() -> None:
    assert color_for("model_b") == MODEL_COLORS["model_b"]
    assert color_for("adam", kind="optimizer") == OPTIMIZER_COLORS["adam"]
    assert color_for("train", kind="split") == SPLIT_COLORS["train"]
    assert color_for("not_a_model") == NEUTRAL_GREY
    with pytest.raises(ValueError):
        color_for("model_b", kind="shape")


def test_label_for_names_models_and_optimizers() -> None:
    assert label_for("model_b") == "Model B"
    assert label_for("squeezenet1_1") == "SqueezeNet 1.1"
    assert label_for("sgd_momentum", kind="optimizer") == "SGD + momentum"
    assert label_for("resnet18") == "resnet18"
    with pytest.raises(ValueError):
        label_for("train", kind="split")


def test_save_figure_writes_and_copies(tmp_path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    annotate_hardware(fig, "cpu (test)")
    saved = save_figure(fig, tmp_path / "figures" / "demo.png", also_copy_to=tmp_path / "report")
    plt.close(fig)
    assert saved.exists() and saved.stat().st_size > 0
    assert (tmp_path / "report" / "demo.png").exists()
