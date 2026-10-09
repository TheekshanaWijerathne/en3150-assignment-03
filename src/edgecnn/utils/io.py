"""Small file helpers: hashing, directories, and Markdown tables for the report.

JSON results are written through ``edgecnn.contracts.schema`` instead, which
validates them first.

Team notes:
Owner: Member 1. markdown_table / write_markdown_table are used by Member 2
(architecture tables) and Member 4 (comparison tables).
"""

from __future__ import annotations

import hashlib
from numbers import Integral, Real
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 65536) -> str:
    """Streaming SHA-256 of a file, as lowercase hex.

    Used for ``split_meta.json -> manifest_sha256``, which is how anyone
    checks that the split on disk is the one the committed results came from.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if absent; return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_relative(path: Path) -> str:
    """``path`` relative to the repository, with forward slashes, when it is inside it.

    For messages and recorded paths: notebook outputs and result files are
    committed, so they must never contain a personal absolute path.
    """
    from edgecnn.contracts.paths import REPO_ROOT

    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def format_cell(value: Any) -> str:
    """Format one table cell: thousands separators for integers, 4 significant
    digits for small floats, one decimal for large ones, ``—`` for missing values."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return f"{int(value):,}"
    if isinstance(value, Real):
        number = float(value)
        return f"{number:,.1f}" if abs(number) >= 1000 else f"{number:.4g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(headers: list[str], rows: list[list[Any]], caption: str | None = None) -> str:
    """A GitHub-flavoured Markdown table as a string.

    Columns whose values are all numbers are right-aligned, so digits line up.

    Args:
        headers: Column names.
        rows: One list of cell values per row, in header order.
        caption: Optional line printed in italics above the table.

    Raises:
        ValueError: if a row's length differs from the number of headers.
    """
    for index, row in enumerate(rows):
        if len(row) != len(headers):
            raise ValueError(
                f"row {index} has {len(row)} cells but there are {len(headers)} headers"
            )

    def numeric(column: int) -> bool:
        cells = [row[column] for row in rows if row[column] is not None]
        return bool(cells) and all(
            isinstance(cell, Real) and not isinstance(cell, bool) for cell in cells
        )

    lines = []
    if caption:
        lines += [f"*{caption}*", ""]
    lines.append("| " + " | ".join(format_cell(h) for h in headers) + " |")
    lines.append("|" + "|".join("---:" if numeric(c) else "---" for c in range(len(headers))) + "|")
    for row in rows:
        lines.append("| " + " | ".join(format_cell(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_table(
    path: Path,
    headers: list[str],
    rows: list[list[Any]],
    caption: str | None = None,
) -> Path:
    """Write :func:`markdown_table` to ``path``, creating parent directories.

    Markdown rather than CSV because the tables are pasted straight into the
    report. Returns ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_table(headers, rows, caption), encoding="utf-8")
    return path
