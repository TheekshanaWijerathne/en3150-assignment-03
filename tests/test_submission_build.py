"""The submission folder: its notebooks build from their sources, match them, and pass the gate."""

from __future__ import annotations

import json
import sys

import pytest

from edgecnn.contracts import paths

nbformat = pytest.importorskip("nbformat", reason="nbformat not installed")

sys.path.insert(0, str(paths.REPO_ROOT / "tools"))
import build_submission as build  # noqa: E402

TEXTS = sorted(build.NOTEBOOK_TEXTS.glob("*.md"))


def test_there_are_seven_notebooks() -> None:
    assert [text.stem[:2] for text in TEXTS] == ["01", "02", "03", "04", "05", "06", "07"]


@pytest.mark.parametrize("text", TEXTS, ids=lambda path: path.stem)
def test_every_notebook_builds_and_passes_the_gate(text) -> None:
    notebook = build.build_notebook(text)  # validates, and fails on a missing part
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "<!-- include" not in source
    assert build.scan(source, text.name) == []


@pytest.mark.parametrize("text", TEXTS, ids=lambda path: path.stem)
def test_committed_notebooks_match_their_sources(text) -> None:
    """The generated notebooks are never edited by hand: a fix goes into the sources."""
    committed = build.TARGET / f"{text.stem}.ipynb"
    if not committed.exists():
        pytest.skip("the submission folder has not been built")
    cells = json.loads(committed.read_text(encoding="utf-8"))["cells"]
    expected = build.build_notebook(text).cells
    assert ["".join(cell["source"]) for cell in cells] == [cell.source for cell in expected], (
        f"{committed.name} differs from its sources: run python tools/build_submission.py"
    )


@pytest.mark.parametrize(
    "text",
    [
        "Loss \u2014 accuracy",
        "see \u00a73",
        "03 \u00b7 Optimizer",
        'MODE = "official"',
        "from google.colab import userdata",
        "Owner: Member 2",
        "written by Theekshana",
        "index 210123A",
        "mail someone@example.com",
        r"C:\Users\x\file.png",
        "/content/repo",
    ],
)
def test_the_gate_catches(text) -> None:
    assert build.scan(text, "probe")


def test_the_gate_allows_the_repository_link_and_ordinary_code() -> None:
    text = (
        f"see <{build.REPOSITORY}> and https://pytorch.org; f'{{value:,.1f}}'; "
        "cuda:0 (Tesla T4); i5-11400H @ 2.70GHz"
    )
    assert build.scan(text, "probe") == []
