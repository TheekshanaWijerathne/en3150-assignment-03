"""Shared figure style.

Owner: Member 1.  Used by Member 3 (curves.py) and Member 4 (reporting.py).

Only style and generic helpers live here. The actual figures belong to whoever
owns the marking section they support - see curves.py and reporting.py.

The point of a shared style module is that figures from three different people
sit on facing pages of one report and should not look like three reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

#: Consistent colour per model across every figure in the report. A reader who
#: learns "orange is Model B" on the loss curves should not have to relearn it
#: on the scatter plot.
MODEL_COLORS: dict[str, str] = {
    "model_a": "#4C72B0",
    "model_b": "#DD8452",
    "mobilenet_v2": "#55A868",
    "squeezenet1_1": "#C44E52",
}

#: Consistent colour per optimizer for the Section 3 overlay.
OPTIMIZER_COLORS: dict[str, str] = {
    "sgd": "#8172B3",
    "sgd_momentum": "#937860",
    "adam": "#DA8BC3",
}

#: Figures go into a printed report, so default DPI is print-quality.
FIGURE_DPI: int = 200


def apply_style() -> None:
    """Set matplotlib rcParams once, at the top of any plotting script.

    Font sizes should stay legible after the figure is scaled into a
    two-column report - a default-sized axis label is usually unreadable at
    half width. Prefer a white background and a light grid; a dark theme wastes
    toner and reproduces badly in print.
    """
    raise NotImplementedError("Member 1: implement apply_style")


def save_figure(fig: Figure, path: Path, also_copy_to: Path | None = None) -> Path:
    """Save at FIGURE_DPI with a tight bounding box, creating parent dirs.

    `also_copy_to` mirrors the figure into report/figures/, so the report
    always pulls from one place.
    """
    raise NotImplementedError("Member 1: implement save_figure")


def color_for(key: str, kind: str = "model") -> str:
    """Look up the consistent colour for a model or optimizer key.

    Falls back to a neutral grey for an unknown key rather than raising - a
    missing colour should not abort a figure.
    """
    raise NotImplementedError("Member 1: implement color_for")


def annotate_hardware(fig: Figure, device: str, **kwargs: Any) -> None:
    """Stamp the device string onto a timing-related figure.

    Section 4 requires the evaluation hardware to be reported, and a figure
    that travels into a slide deck should carry its own provenance.
    """
    raise NotImplementedError("Member 1: implement annotate_hardware")
