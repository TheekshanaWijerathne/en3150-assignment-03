"""Build the submission folder, and check it.

    python tools/build_submission.py            write the notebooks, README.md and requirements.txt
    python tools/build_submission.py --docs     rewrite only README.md and requirements.txt
    python tools/build_submission.py --check    check the folder as it is, outputs included

The notebooks are generated from the notebook texts in tools/submission/notebooks/, which include
shared implementation parts from tools/submission/parts/: every notebook that uses a part gets the
same copy, so the copies cannot drift apart. A notebook that already has outputs is overwritten only
with --force, so an executed run is never lost by accident.

README.md is filled in from the executed notebooks: the environment lines of notebook 03 (training)
and notebook 06 (latency).

Everything written, and with --check every output too, must pass the gate: no em-dash, section sign
or middle dot; no member name, username, index number or email; no development-only feature; no
machine path.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "submission"
PARTS = SOURCE / "parts"
NOTEBOOK_TEXTS = SOURCE / "notebooks"
TARGET = ROOT / "submission" / "Misclassified_A03_EN3150"
REPOSITORY = "https://github.com/TheekshanaWijerathne/en3150-assignment-03"

FENCE = re.compile(r"^```python\s*$")
INCLUDE = re.compile(r"^<!--\s*include:\s*([\w-]+)\s*-->\s*$")
KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}

# --- the gate -----------------------------------------------------------------------------

BANNED_CHARACTERS = {"—": "em-dash", "§": "section sign", "·": "middle dot"}
DEVELOPMENT_TERMS = [
    r"\bMODE\b",
    r"synthetic",
    r"is_official",
    r"autoreload",
    r"google\.colab",
    r"\bIN_COLAB\b",
    r"push_results",
    r"edgecnn",
    r"Team notes",
    r"#\s*team:",
    r"\bTODO\b",
    r"NotImplementedError",
    r"\bseams?\b",
    r"CONTRIBUTING",
    r"COLAB\.md",
    r"SUBMISSION\.md",
    r"\bMember\s*\d",
    r"\bmember_\d",
    r"\bM[1-4]\b",
    r"GH_TOKEN",
    r"A03_REPO",
    r"A03_BRANCH",
]
PERSONAL_PATTERNS = [
    (r"\b\d{6}[A-Za-z]\b", "an index number"),
    (r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}", "an email address"),
    (r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,}", "a GitHub token"),
]
PATH_PATTERNS = [
    (r"(?<![A-Za-z])[A-Za-z]:[\\/](?![\\/])", "a drive-letter path"),
    (r"/home/|/Users/|/content/|/root/", "a machine path"),
]


def _denylist() -> re.Pattern[str]:
    words = [
        line.strip()
        for line in (SOURCE / "denylist.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)


def scan(text: str, where: str) -> list[str]:
    """Every gate violation in ``text``, described with ``where`` it was found."""
    text = text.replace(REPOSITORY, "")  # the repository link is wanted, username and all
    hits = []

    def add(match: re.Match[str], reason: str) -> None:
        start, end = max(0, match.start() - 30), min(len(text), match.end() + 30)
        context = text[start:end].replace("\n", " ")
        hits.append(f"{where}: {reason} {match.group(0)!r} in ...{context}...")

    for character, name in BANNED_CHARACTERS.items():
        for match in re.finditer(re.escape(character), text):
            add(match, f"a banned character ({name})")
    for pattern in DEVELOPMENT_TERMS:
        for match in re.finditer(pattern, text):
            add(match, "development-only text")
    for match in _denylist().finditer(text):
        add(match, "a member detail")
    for pattern, reason in PERSONAL_PATTERNS + PATH_PATTERNS:
        for match in re.finditer(pattern, text):
            add(match, reason)
    return hits


def _output_text(output: dict) -> str:
    """The readable text of one cell output; images and widget state are skipped."""
    if output.get("output_type") == "stream":
        return "".join(output.get("text", ""))
    if output.get("output_type") == "error":
        return "\n".join(
            [output.get("ename", ""), output.get("evalue", "")] + output.get("traceback", [])
        )
    data = output.get("data", {})
    parts = []
    for kind in ("text/plain", "text/markdown", "text/html"):
        value = data.get(kind)
        if value:
            parts.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(parts)


def scan_notebook(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    hits = scan(json.dumps(notebook.get("metadata", {})), f"{path.name} metadata")
    for index, cell in enumerate(notebook["cells"]):
        where = f"{path.name} cell {index} ({cell.get('id')})"
        hits += scan("".join(cell.get("source", "")), where)
        for output in cell.get("outputs", []):
            hits += scan(_output_text(output), f"{where} output")
    return hits


def scan_folder(folder: Path) -> list[str]:
    """Every gate violation in the folder's text files; data/ and checkpoints/ are not submitted."""
    hits = []
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        if not path.is_file() or relative.parts[0] in ("data", "checkpoints"):
            continue
        if path.suffix == ".ipynb":
            hits += scan_notebook(path)
        elif path.suffix in (".md", ".txt", ".json", ".csv"):
            hits += scan(path.read_text(encoding="utf-8"), str(relative))
    return hits


# --- building ----------------------------------------------------------------------------


def parse(path: Path) -> list[tuple[str, str]]:
    """Cells of a notebook text: ``python`` code fences become code cells, the text between
    them markdown cells, and ``<!-- include: name -->`` inserts the cells of a shared part."""
    cells: list[tuple[str, str]] = []
    text: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()

    def flush() -> None:
        markdown = "\n".join(text).strip("\n")
        if markdown.strip():
            cells.append(("markdown", markdown))
        text.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        if FENCE.match(line):
            flush()
            end = index + 1
            while end < len(lines) and lines[end].rstrip() != "```":
                end += 1
            if end == len(lines):
                raise ValueError(f"{path}: unclosed code block from line {index + 1}")
            cells.append(("code", "\n".join(lines[index + 1 : end])))
            index = end + 1
            continue
        include = INCLUDE.match(line)
        if include:
            flush()
            cells += parse(PARTS / f"{include.group(1)}.md")
        else:
            text.append(line)
        index += 1
    flush()
    return cells


def build_notebook(text_path: Path) -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {"kernelspec": KERNELSPEC, "language_info": {"name": "python"}}
    prefix = text_path.stem[:2]
    for index, (kind, source) in enumerate(parse(text_path)):
        cell_id = f"s{prefix}-{index:02d}"
        maker = nbformat.v4.new_markdown_cell if kind == "markdown" else nbformat.v4.new_code_cell
        notebook.cells.append(maker(source, id=cell_id))
    nbformat.validate(notebook)
    return notebook


def has_outputs(path: Path) -> bool:
    if not path.exists():
        return False
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return any(cell.get("outputs") for cell in notebook["cells"])


def write_notebooks(force: bool) -> list[Path]:
    written = []
    for text_path in sorted(NOTEBOOK_TEXTS.glob("*.md")):
        target = TARGET / f"{text_path.stem}.ipynb"
        if has_outputs(target) and not force:
            print(f"kept {target.name}: it has outputs from a run (rebuild it with --force)")
            continue
        text = nbformat.writes(build_notebook(text_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            text if text.endswith("\n") else text + "\n", encoding="utf-8", newline="\n"
        )
        written.append(target)
    return written


def environment(notebook: Path) -> str | None:
    """The device and versions a notebook printed in its setup cell, if it has run."""
    if not notebook.exists():
        return None
    values: dict[str, str] = {}
    for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]:
        for output in cell.get("outputs", []):
            for line in _output_text(output).splitlines():
                match = re.match(r"^(Python|PyTorch|torchvision|NumPy|Device)\s{2,}(.+)$", line)
                if match:
                    values.setdefault(match.group(1), match.group(2).strip())
    if "Device" not in values:
        return None
    versions = ", ".join(
        f"{name} {values[name]}"
        for name in ("Python", "PyTorch", "torchvision", "NumPy")
        if name in values
    )
    return f"{values['Device']}; {versions}"


def write_docs() -> list[Path]:
    readme = (SOURCE / "README.md").read_text(encoding="utf-8")
    pending = "not recorded yet (the notebook has not been run)"
    readme = (
        readme.replace("{{repository}}", REPOSITORY)
        .replace(
            "{{training_environment}}", environment(TARGET / "03_optimizer_study.ipynb") or pending
        )
        .replace(
            "{{benchmark_environment}}",
            environment(TARGET / "06_resource_benchmark.ipynb") or pending,
        )
    )
    targets = [TARGET / "README.md", TARGET / "requirements.txt"]
    TARGET.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(readme, encoding="utf-8", newline="\n")
    targets[1].write_text(
        (SOURCE / "requirements.txt").read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
    )
    return targets


def status(folder: Path) -> list[str]:
    """Which notebooks have run completely, and which still need a run."""
    lines = []
    for path in sorted(folder.glob("*.ipynb")):
        cells = [
            c
            for c in json.loads(path.read_text(encoding="utf-8"))["cells"]
            if c["cell_type"] == "code"
        ]
        ran = sum(1 for c in cells if c.get("execution_count"))
        errors = sum(
            1 for c in cells for o in c.get("outputs", []) if o.get("output_type") == "error"
        )
        state = "ran" if ran == len(cells) and not errors else f"{ran}/{len(cells)} cells ran"
        lines.append(f"  {path.name:<40} {state}{f', {errors} error(s)' if errors else ''}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--docs", action="store_true", help="rewrite only README.md and requirements.txt"
    )
    parser.add_argument(
        "--check", action="store_true", help="only check the folder, outputs included"
    )
    parser.add_argument(
        "--force", action="store_true", help="also overwrite notebooks that have outputs"
    )
    args = parser.parse_args()

    if not args.check:
        written = [] if args.docs else write_notebooks(args.force)
        written += write_docs()
        for path in written:
            print(f"wrote {path.relative_to(ROOT).as_posix()}")

    hits = scan_folder(TARGET)
    print("\n".join(status(TARGET)))
    if hits:
        print(f"\nThe gate found {len(hits)} problem(s):")
        print("\n".join(f"  {hit}" for hit in hits))
        return 1
    print("\nThe gate passed: no banned character, member detail, development feature or path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
