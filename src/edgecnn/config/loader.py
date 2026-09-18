"""Layered YAML configuration.

Owner: Member 1. Consumed by every script and every member.

The layering
------------
A run's configuration is assembled from three files, each overriding the last::

    configs/base.yaml              shared by everything: seed, device, paths
        <- configs/stages/*.yaml   one per pipeline stage, declares inputs/outputs
            <- configs/experiments/<name>.yaml   the specific run

An experiment config names the stages it pulls in via a top-level ``extends``
list, then overrides whatever it needs. That keeps the six experiment files
short enough to read at a glance and means a change to, say, the augmentation
policy happens in one place rather than six.

Why stage configs declare their own inputs and outputs
------------------------------------------------------
Every ``configs/stages/*.yaml`` has explicit ``inputs:`` and ``outputs:``
blocks. That is the seam made visible: you can see what a stage consumes and
what it hands on without reading a line of Python, and the owner of the next
stage downstream can read it without asking anyone.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from edgecnn.contracts.paths import CONFIGS_DIR, REPO_ROOT, make_run_id
from edgecnn.contracts.types import DEFAULT_SEED, ResolvedConfig


class ConfigError(Exception):
    """Raised for a malformed, missing or contradictory configuration."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(loaded).__name__}")
    return loaded


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, returning a new dict.

    Nested mappings merge key by key; every other type (including lists) is
    replaced wholesale. Lists are replaced rather than concatenated on
    purpose - an experiment that sets ``augmentation: [random_flip]`` means
    *only* that, not "the base list plus this".
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_path(value: str | Path) -> Path:
    """Turn a repo-relative config path into an absolute one.

    Config files always spell paths relative to the repository root, so the
    same YAML works from ``scripts/``, ``tests/`` and ``notebooks/`` without
    anyone thinking about the current working directory.
    """
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_stage(name: str) -> dict[str, Any]:
    """Load one ``configs/stages/<name>.yaml``."""
    return _read_yaml(CONFIGS_DIR / "stages" / f"{name}.yaml")


def load_config(path: str | Path, **overrides: Any) -> ResolvedConfig:
    """Compose base + stages + experiment into a :class:`ResolvedConfig`.

    Args:
        path: An experiment config, normally under ``configs/experiments/``.
        **overrides: Last-word CLI overrides, e.g. ``seed=1``. Top-level keys
            only; use a dotted path for nesting, e.g.
            ``load_config(p, **{"training.epochs": 2})``.

    Raises:
        ConfigError: if a required key is missing, or if ``run_id`` is stated
            in the file and disagrees with model/optimizer/seed - a mismatch
            there would scatter one run's artifacts across two directories.
    """
    experiment_path = resolve_path(path)
    experiment = _read_yaml(experiment_path)

    config: dict[str, Any] = _read_yaml(CONFIGS_DIR / "base.yaml")

    for stage_name in experiment.get("extends", []):
        config = merge(config, load_stage(stage_name))

    config = merge(config, experiment)
    config.pop("extends", None)

    for dotted, value in overrides.items():
        _set_dotted(config, dotted, value)

    model_name = _require(config, "model", "name", context=experiment_path)
    optimizer_name = _require(config, "optimizer", "name", context=experiment_path)
    seed = int(config.get("seed", DEFAULT_SEED))

    run_id = make_run_id(model_name, optimizer_name, seed)
    declared = config.get("run_id")
    if declared is not None and declared != run_id:
        raise ConfigError(
            f"{experiment_path}: run_id is declared as {declared!r} but "
            f"model/optimizer/seed imply {run_id!r}. Artifacts would be written "
            f"to one directory and read from another. Fix one or the other."
        )
    config["run_id"] = run_id

    return ResolvedConfig(
        raw=config,
        run_id=run_id,
        model_name=model_name,
        optimizer_name=optimizer_name,
        seed=seed,
        device=str(config.get("device", "auto")),
    )


def _require(config: dict[str, Any], section: str, key: str, *, context: Path) -> str:
    block = config.get(section)
    if not isinstance(block, dict) or key not in block:
        raise ConfigError(f"{context}: missing required key '{section}.{key}'")
    return str(block[key])


def _set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor = config
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value
