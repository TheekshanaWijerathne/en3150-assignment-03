"""Runtime enforcement of the on-disk seam formats.

    +---------------------------------------------------------------------+
    |  FROZEN INTERFACE.  PR + 1 review to change.                         |
    +---------------------------------------------------------------------+

The JSON Schemas in ``configs/contracts/`` describe every file that crosses
between members. This module turns those documents from documentation into
something that fails loudly.

How to use it
-------------
*Producers* call :func:`write_json` instead of ``json.dump``. The payload is
validated before it touches disk, so a malformed artifact is never written::

    from edgecnn.contracts import schema, paths
    schema.write_json(paths.history_json(run_id), payload, "history")

*Consumers* call :func:`read_json`, which validates on the way in. If someone
hand-edited a file or an older run predates a schema change, you find out at
the read, not three functions later inside a plotting call::

    history = schema.read_json(paths.history_json(run_id), "history")

If validation fails, that is a broken interface - not a bug in your code.
Tell whoever owns the producing stage; do not work around it locally.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from edgecnn.contracts.paths import SCHEMAS_DIR

#: Schema name -> filename in configs/contracts/.
SCHEMA_FILES: dict[str, str] = {
    "split_manifest": "split_manifest.schema.json",
    "split_meta": "split_meta.schema.json",
    "norm_stats": "norm_stats.schema.json",
    "history": "history.schema.json",
    "test_metrics": "test_metrics.schema.json",
    "resources": "resources.schema.json",
}

#: Required column order of data/splits/split_manifest.csv.
#: CSV is not JSON, so the header is checked here rather than by jsonschema;
#: the row *values* are still validated against split_manifest.schema.json.
MANIFEST_COLUMNS: tuple[str, ...] = ("relative_path", "label_index", "label_name", "split")

_CACHE: dict[str, dict[str, Any]] = {}


class ContractViolation(Exception):
    """Raised when an artifact does not match its declared schema.

    Deliberately not a ``ValueError``. Seeing this name in a traceback should
    immediately tell you the problem is a cross-member interface, not local
    logic.
    """


def load_schema(name: str) -> dict[str, Any]:
    """Load and cache a JSON Schema document by short name."""
    if name in _CACHE:
        return _CACHE[name]
    if name not in SCHEMA_FILES:
        raise KeyError(f"unknown schema {name!r}; expected one of {sorted(SCHEMA_FILES)}")
    path = SCHEMAS_DIR / SCHEMA_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"schema document missing: {path}")
    _CACHE[name] = json.loads(path.read_text(encoding="utf-8"))
    return _CACHE[name]


def validate(payload: Any, schema_name: str, *, source: str | Path = "<memory>") -> None:
    """Validate ``payload`` against a named schema.

    Raises:
        ContractViolation: with the failing JSON path and the expectation.
    """
    import jsonschema

    try:
        jsonschema.validate(instance=payload, schema=load_schema(schema_name))
    except jsonschema.ValidationError as exc:
        location = "/".join(str(part) for part in exc.absolute_path) or "<root>"
        raise ContractViolation(
            f"{source}: does not satisfy the {schema_name!r} contract.\n"
            f"  at      : {location}\n"
            f"  problem : {exc.message}\n"
            f"  schema  : {SCHEMAS_DIR / SCHEMA_FILES[schema_name]}\n"
            f"  This is an interface break. Tell the member who owns this stage."
        ) from exc


def write_json(path: Path, payload: Any, schema_name: str) -> Path:
    """Validate, then write pretty-printed JSON. Creates parent directories."""
    validate(payload, schema_name, source=path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path, schema_name: str) -> Any:
    """Read JSON and validate it before handing it back."""
    if not path.exists():
        raise FileNotFoundError(
            f"expected artifact missing: {path}\n"
            f"  Has the producing stage been run for this run_id?"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate(payload, schema_name, source=path)
    return payload


def validate_manifest(path: Path) -> int:
    """Validate ``split_manifest.csv`` header and every row.

    Returns:
        Number of data rows.

    Raises:
        ContractViolation: on a wrong header, an illegal ``split`` value, a
            non-integer ``label_index``, or a backslash in ``relative_path``
            (paths are stored POSIX-style so the manifest is identical on
            Windows and Linux machines - the team runs on both).
    """
    if not path.exists():
        raise FileNotFoundError(f"split manifest missing: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != MANIFEST_COLUMNS:
            raise ContractViolation(
                f"{path}: header is {header}, expected {MANIFEST_COLUMNS}"
            )
        count = 0
        for line_no, row in enumerate(reader, start=2):
            if "\\" in row["relative_path"]:
                raise ContractViolation(
                    f"{path}:{line_no}: relative_path must use forward slashes, "
                    f"got {row['relative_path']!r}"
                )
            validate(
                {
                    "relative_path": row["relative_path"],
                    "label_index": int(row["label_index"]),
                    "label_name": row["label_name"],
                    "split": row["split"],
                },
                "split_manifest",
                source=f"{path}:{line_no}",
            )
            count += 1

    if count == 0:
        raise ContractViolation(f"{path}: manifest has a header but no rows")
    return count


def validate_checkpoint(path: Path) -> None:
    """Check that a checkpoint carries every key in ``CHECKPOINT_KEYS``.

    Torch checkpoints are pickles, not JSON, so this is a key-set check rather
    than a schema validation. Member 1 needs ``class_names`` and
    ``input_shape`` out of the checkpoint to evaluate a run without re-reading
    the config, which is why they are mandatory.
    """
    import torch

    from edgecnn.contracts.types import CHECKPOINT_KEYS

    if not path.exists():
        raise FileNotFoundError(f"checkpoint missing: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ContractViolation(f"{path}: expected a dict, got {type(blob).__name__}")
    missing = [key for key in CHECKPOINT_KEYS if key not in blob]
    if missing:
        raise ContractViolation(
            f"{path}: checkpoint is missing required keys {missing}.\n"
            f"  Required: {list(CHECKPOINT_KEYS)}\n"
            f"  Found   : {sorted(blob)}"
        )
