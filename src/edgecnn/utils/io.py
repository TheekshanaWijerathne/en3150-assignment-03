"""Small filesystem helpers.

Owner: Member 1.

For JSON that crosses a member boundary use
``edgecnn.contracts.schema.read_json`` / ``write_json`` instead - those
validate against the contract. These helpers are for everything else.
"""

from __future__ import annotations

import hashlib
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


def write_markdown_table(
    path: Path,
    headers: list[str],
    rows: list[list[Any]],
    caption: str | None = None,
) -> Path:
    """Write a GitHub-flavoured Markdown table.

    Used by Member 4 for everything in ``results/tables/``. Markdown rather
    than CSV because these are pasted straight into the report.
    """
    raise NotImplementedError("Member 1: implement write_markdown_table")
