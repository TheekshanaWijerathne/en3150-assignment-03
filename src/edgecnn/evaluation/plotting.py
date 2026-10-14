"""Shared figure style, so every figure in the report looks like it belongs to one report.

Figures are drawn at two resolutions on purpose. Inline figures are stored
inside the notebooks, so they render at ``INLINE_DPI`` to keep the notebooks
small; the PNG files used in the report are saved at ``FIGURE_DPI``.

Team notes:
Owner: Member 1. Only style and generic helpers live here - the figures
themselves belong to curves.py (Member 3) and reporting.py (Member 4).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: One colour per model across every figure: a reader who learns "orange is
#: Model B" on the loss curves should not have to relearn it on the scatter plot.
MODEL_COLORS: dict[str, str] = {
    "model_a": "#4C72B0",
    "model_b": "#DD8452",
    "mobilenet_v2": "#55A868",
    "squeezenet1_1": "#C44E52",
}

#: One colour per optimizer, for the optimizer-comparison figures.
OPTIMIZER_COLORS: dict[str, str] = {
    "sgd": "#8172B3",
    "sgd_momentum": "#937860",
    "adam": "#DA8BC3",
}

#: How each model and optimizer is named in figures and tables.
MODEL_LABELS: dict[str, str] = {
    "model_a": "Model A",
    "model_b": "Model B",
    "mobilenet_v2": "MobileNetV2",
    "squeezenet1_1": "SqueezeNet 1.1",
}
OPTIMIZER_LABELS: dict[str, str] = {
    "sgd": "SGD",
    "sgd_momentum": "SGD + momentum",
    "adam": "Adam",
}

#: One colour per split, one hue from dark to light, so the three bars of a
#: class read as one group in the dataset figures.
SPLIT_COLORS: dict[str, str] = {
    "train": "#2F5E8E",
    "val": "#6C9BCF",
    "test": "#B3CDE8",
}

#: Inline (notebook) resolution - keeps committed notebooks small.
INLINE_DPI: int = 100

#: Saved-PNG resolution - the figures go into a printed report.
FIGURE_DPI: int = 200

#: Colour for anything without a fixed colour.
NEUTRAL_GREY: str = "#7F7F7F"

_STYLE: dict[str, Any] = {
    "figure.dpi": INLINE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
    "figure.figsize": (6.0, 4.0),
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    # Sizes chosen to stay legible when a figure is scaled to half a page width.
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
}


def apply_style() -> None:
    """Apply the shared matplotlib style. Call once per notebook, before plotting.

    Sets inline figures to ``INLINE_DPI`` and saved files to ``FIGURE_DPI``, with
    font sizes that remain legible at half page width, a white background and a
    light grid - a dark theme wastes toner and reproduces badly in print.

    In a notebook it also turns on inline display, so a function that returns a
    ``Figure`` shows it once, as an image. Build figures with
    ``matplotlib.figure.Figure`` and return them; a figure made with pyplot must
    be closed before it is returned, or the notebook shows it twice.
    """
    import matplotlib.pyplot as plt

    _enable_inline_display()  # first: activating the backend may reset settings
    plt.rcParams.update(_STYLE)


def _enable_inline_display() -> None:
    """In a Jupyter kernel, show returned figures inline, whether or not pyplot made them."""
    try:
        from IPython import get_ipython
    except ImportError:
        return
    shell = get_ipython()
    if shell is not None and getattr(shell, "kernel", None) is not None:
        shell.run_line_magic("matplotlib", "inline")


def save_figure(fig: Figure, path: Path, also_copy_to: Path | None = None) -> Path:
    """Save ``fig`` as a print-quality PNG, creating parent directories.

    Args:
        fig: The figure to save.
        path: Destination file.
        also_copy_to: Optional folder (or file path) that receives a copy, e.g.
            ``report/figures/``, so the report always pulls from one place.

    Returns:
        ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    if also_copy_to is not None:
        target = Path(also_copy_to)
        if target.suffix == "":
            target = target / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return path


def color_for(key: str, kind: str = "model") -> str:
    """The fixed colour for a model, optimizer or split key.

    An unknown key gets a neutral grey rather than an error - a missing colour
    should never abort a figure.

    Raises:
        ValueError: if ``kind`` is not ``"model"``, ``"optimizer"`` or ``"split"``.
    """
    palettes = {"model": MODEL_COLORS, "optimizer": OPTIMIZER_COLORS, "split": SPLIT_COLORS}
    if kind not in palettes:
        raise ValueError(f"kind must be one of {sorted(palettes)}, not {kind!r}")
    return palettes[kind].get(key, NEUTRAL_GREY)


def label_for(key: str, kind: str = "model") -> str:
    """The display name for a model or optimizer key, such as ``"Model B"`` for ``"model_b"``.

    An unknown key is shown as it is.

    Raises:
        ValueError: if ``kind`` is not ``"model"`` or ``"optimizer"``.
    """
    labels = {"model": MODEL_LABELS, "optimizer": OPTIMIZER_LABELS}
    if kind not in labels:
        raise ValueError(f"kind must be one of {sorted(labels)}, not {kind!r}")
    return labels[kind].get(key, key)


def annotate_hardware(fig: Figure, device: str, **kwargs: Any) -> None:
    """Stamp the device string in small grey text at the figure's bottom-right corner.

    Timing figures should carry the hardware they were measured on, so they stay
    meaningful when copied into slides or the report. Extra keyword arguments go
    to ``Figure.text``.
    """
    style = {"ha": "right", "va": "bottom", "fontsize": 7, "color": "#666666"}
    style.update(kwargs)
    fig.text(0.99, 0.01, device, **style)
