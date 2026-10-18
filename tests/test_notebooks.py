"""The section notebooks exist, are well-formed, and follow the standard layout.

Notebooks are checked for STRUCTURE only - running them needs data and trained
models. What is checked is what keeps the workflow safe:

* the eight section notebooks exist under their fixed names;
* each is valid nbformat-4 JSON with unique cell ids;
* cell 1 is the guarded Colab bootstrap, so it does nothing locally;
* cell 2 sets MODE, so every notebook can run in synthetic / debug / official;
* no notebook patches sys.path outside the Colab guard - locally the editable
  install is the only import mechanism;
* no notebook ever contains a GitHub token.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from edgecnn.contracts import paths

SECTION_NOTEBOOKS = (
    "00_setup",
    "01_data_preparation",
    "02_custom_architectures",
    "03_optimizer_study",
    "04_custom_training_evaluation",
    "05_pretrained_finetuning",
    "06_resource_benchmark",
    "07_final_comparison",
)

#: GitHub token shapes: classic (ghp_...), fine-grained (github_pat_...), OAuth/app.
TOKEN_PATTERN = re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}")


def _load(stem: str) -> dict:
    return json.loads((paths.NOTEBOOKS_DIR / f"{stem}.ipynb").read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def test_every_section_notebook_exists() -> None:
    missing = [s for s in SECTION_NOTEBOOKS if not (paths.NOTEBOOKS_DIR / f"{s}.ipynb").exists()]
    assert not missing, f"missing section notebooks: {missing}"


@pytest.mark.parametrize("stem", SECTION_NOTEBOOKS)
def test_notebook_is_valid_nbformat4(stem: str) -> None:
    notebook = _load(stem)
    assert notebook["nbformat"] == 4
    metadata = notebook["metadata"]
    # Colab saves leave out kernelspec.language (it is optional); language_info names it.
    language = metadata["kernelspec"].get("language") or metadata["language_info"]["name"]
    assert language == "python"
    ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(ids) and len(ids) == len(set(ids)), f"{stem}: cell ids missing or duplicated"


@pytest.mark.parametrize("stem", SECTION_NOTEBOOKS)
def test_first_cell_is_the_guarded_colab_bootstrap(stem: str) -> None:
    first = _code_cells(_load(stem))[0]
    # "installed", not "already imported": also true on a runtime reached from VS Code
    assert 'find_spec("google.colab")' in first
    assert "if IN_COLAB:" in first, f"{stem}: the bootstrap must do nothing locally"


@pytest.mark.parametrize("stem", SECTION_NOTEBOOKS)
def test_bootstrap_refreshes_the_clone_on_every_rerun(stem: str) -> None:
    """Pushing a fix mid-session must reach the running Colab machine."""
    first = _code_cells(_load(stem))[0]
    assert '"fetch"' in first and '"-B", BRANCH' in first, (
        f"{stem}: re-running cell 1 must fetch and check out the latest REPO@BRANCH"
    )


@pytest.mark.parametrize("stem", SECTION_NOTEBOOKS)
def test_bootstrap_reads_personal_settings_from_colab_secrets(stem: str) -> None:
    """Each member's fork / branch comes from their own Colab Secrets, never the notebook."""
    first = _code_cells(_load(stem))[0]
    assert '_secret("A03_REPO"' in first and '_secret("A03_BRANCH"' in first
    assert "except Exception" in first, f"{stem}: unreadable Secrets (VS Code) must fall back"


def test_bootstrap_defaults_are_not_personal() -> None:
    """In VS Code the defaults are edited for a session. Committing your own fork or
    branch would change everyone's default, so every notebook must carry the same values."""
    settings = {}
    for stem in SECTION_NOTEBOOKS:
        first = _code_cells(_load(stem))[0]
        repo = re.search(r'REPO = _secret\("A03_REPO", "([^"]+)"\)', first)
        branch = re.search(r'BRANCH = _secret\("A03_BRANCH", "([^"]+)"\)', first)
        assert repo and branch, f"{stem}: cell 1 must define REPO and BRANCH through _secret"
        settings[stem] = (repo.group(1), branch.group(1))
    assert len(set(settings.values())) == 1, f"cell 1 defaults differ between notebooks: {settings}"
    assert next(iter(settings.values()))[1] == "main", "put BRANCH's default back to \"main\" before committing"


@pytest.mark.parametrize("stem", SECTION_NOTEBOOKS)
def test_second_cell_sets_mode_and_autoreload(stem: str) -> None:
    setup = _code_cells(_load(stem))[1]
    assert "%autoreload 2" in setup, f"{stem}: edits to src/ must apply without a restart"
    match = re.search(r'^MODE = "(\w+)"', setup, re.MULTILINE)
    assert match, f"{stem}: setup cell must define MODE"
    assert match.group(1) in ("synthetic", "debug", "official")


@pytest.mark.parametrize("stem", SECTION_NOTEBOOKS)
def test_code_cells_are_valid_python(stem: str) -> None:
    for index, source in enumerate(_code_cells(_load(stem))):
        # A magic becomes `pass`, not nothing: cell 2 runs its magics inside a try block.
        without_magics = "\n".join(
            line[: len(line) - len(line.lstrip())] + "pass"
            if line.lstrip().startswith(("%", "!"))
            else line
            for line in source.splitlines()
        )
        try:
            ast.parse(without_magics)
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{stem} code cell {index}: {exc}")


@pytest.mark.parametrize("stem", SECTION_NOTEBOOKS)
def test_no_sys_path_hacks_outside_the_colab_guard(stem: str) -> None:
    """Locally, imports work through the editable install - never through sys.path."""
    for source in _code_cells(_load(stem))[1:]:
        assert "sys.path" not in source, f"{stem}: remove the sys.path manipulation"


def test_no_notebook_contains_a_github_token() -> None:
    """The repo is public and outputs are committed - a token here would be published."""
    for notebook in sorted(paths.NOTEBOOKS_DIR.rglob("*.ipynb")):
        text = notebook.read_text(encoding="utf-8")
        assert not TOKEN_PATTERN.search(text), f"{notebook.name} contains what looks like a GitHub token"


def test_scratch_folder_exists() -> None:
    assert (paths.NOTEBOOKS_DIR / "scratch" / "README.md").exists()


def test_no_notebook_embeds_an_absolute_user_path() -> None:
    """Outputs are committed; C:\\Users\\<name>\\... paths leak and differ per machine.

    Any drive-letter path counts, such as a ``WindowsPath('C:/...')`` shown as a cell's value.
    """
    pattern = re.compile(
        r"[A-Za-z]:\\\\Users\\\\|/home/[a-z_][a-z0-9_-]*/|/Users/[A-Za-z]"
        r"|(?<![A-Za-z])[A-Za-z]:(?:\\\\|/)(?!/)"
    )
    for notebook in sorted(Path(paths.NOTEBOOKS_DIR).rglob("*.ipynb")):
        text = notebook.read_text(encoding="utf-8")
        assert not pattern.search(text), f"{notebook.name}: print repo-relative paths only"
