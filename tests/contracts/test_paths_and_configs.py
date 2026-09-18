"""run_id derivation and config composition agree across all four members.

If these fail, Member 3 is writing artifacts to a directory Member 1 and
Member 4 are not reading from - which fails silently, because a glob that
matches nothing raises nothing.
"""

from __future__ import annotations

import pytest

from edgecnn.config.loader import ConfigError, load_config, merge
from edgecnn.contracts import paths

pytestmark = pytest.mark.contract


# --- run_id ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "optimizer", "seed", "expected"),
    [
        ("model_a", "adam", 42, "model_a__adam__seed42"),
        ("model_b", "sgd_momentum", 42, "model_b__sgd_momentum__seed42"),
        ("mobilenet_v2", "adam", 42, "mobilenet_v2__adam__seed42"),
        ("squeezenet1_1", "adam", 7, "squeezenet1_1__adam__seed7"),
    ],
)
def test_make_run_id(model: str, optimizer: str, seed: int, expected: str) -> None:
    assert paths.make_run_id(model, optimizer, seed) == expected


@pytest.mark.parametrize(
    "run_id",
    [
        "model_a__adam__seed42",
        "model_b__sgd_momentum__seed42",
        "mobilenet_v2__adam__seed42",
        "squeezenet1_1__adam__seed7",
    ],
)
def test_run_id_round_trips(run_id: str) -> None:
    """Underscores inside a component must not break the split on '__'."""
    assert paths.make_run_id(*paths.parse_run_id(run_id)) == run_id


def test_make_run_id_rejects_double_underscore() -> None:
    with pytest.raises(ValueError, match="double underscore"):
        paths.make_run_id("model__a", "adam", 42)


def test_make_run_id_rejects_uppercase() -> None:
    with pytest.raises(ValueError, match="lowercase"):
        paths.make_run_id("ModelB", "adam", 42)


def test_parse_run_id_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        paths.parse_run_id("not-a-run-id")


# --- path derivation -------------------------------------------------------


def test_repo_root_is_the_repository() -> None:
    """paths.REPO_ROOT is resolved from __file__, so it must not drift."""
    assert (paths.REPO_ROOT / "pyproject.toml").exists()
    assert (paths.REPO_ROOT / "configs" / "base.yaml").exists()


def test_every_run_artifact_lives_under_its_run_dir() -> None:
    """Producers and consumers must agree on the layout, not just the name."""
    run_id = "model_b__adam__seed42"
    assert paths.best_checkpoint(run_id).parent == paths.checkpoint_dir(run_id)
    for artifact in (
        paths.history_json(run_id),
        paths.test_metrics_json(run_id),
        paths.resources_json(run_id),
    ):
        assert artifact.parent == paths.metrics_dir(run_id)
        assert artifact.parent.name == run_id


def test_discover_runs_ignores_non_run_directories() -> None:
    """A stray folder in results/metrics must not become a table row."""
    for run_id in paths.discover_runs():
        paths.parse_run_id(run_id)  # raises if malformed


# --- config composition ----------------------------------------------------


def test_merge_is_recursive_and_non_mutating() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 99}}
    result = merge(base, override)
    assert result == {"a": {"x": 1, "y": 99}, "b": 3}
    assert base["a"]["y"] == 2, "merge must not mutate its input"


def test_merge_replaces_lists_wholesale() -> None:
    """An experiment naming augmentations means exactly those, not base + those."""
    result = merge({"aug": ["flip", "rotate"]}, {"aug": ["flip"]})
    assert result["aug"] == ["flip"]


def test_every_experiment_config_loads(experiment_configs: list) -> None:
    assert experiment_configs, "no experiment configs found"
    for path in experiment_configs:
        cfg = load_config(path)
        assert cfg.run_id
        assert cfg.model_name
        assert cfg.optimizer_name


def test_experiment_filename_matches_derived_run_id(experiment_configs: list) -> None:
    """configs/experiments/model_b__adam.yaml must produce model_b__adam__seed42.

    A mismatch scatters one logical run across two directory names.
    """
    for path in experiment_configs:
        cfg = load_config(path)
        assert cfg.run_id.startswith(path.stem + "__seed"), (
            f"{path.name} produces run_id {cfg.run_id!r}"
        )


def test_every_required_run_has_a_config(experiment_configs: list) -> None:
    """The six runs the report needs must each be reproducible from a config."""
    required = {
        "model_a__adam",
        "model_b__adam",
        "model_b__sgd",
        "model_b__sgd_momentum",
        "mobilenet_v2__adam",
        "squeezenet1_1__adam",
    }
    present = {path.stem for path in experiment_configs}
    assert required <= present, f"missing experiment configs: {sorted(required - present)}"


def test_all_experiments_share_one_seed(experiment_configs: list) -> None:
    """Different seeds across runs would make the comparison table incoherent."""
    seeds = {load_config(path).seed for path in experiment_configs}
    assert len(seeds) == 1, f"experiments disagree on seed: {seeds}"


def test_experiments_meet_the_20_epoch_minimum(experiment_configs: list) -> None:
    for path in experiment_configs:
        cfg = load_config(path)
        epochs = cfg.section("training").get("epochs")
        assert epochs is not None and epochs >= 20, (
            f"{path.name}: epochs={epochs}, assignment requires at least 20"
        )


def test_subset_fraction_is_full_for_committed_runs(experiment_configs: list) -> None:
    """A committed experiment must describe a full-data run."""
    for path in experiment_configs:
        assert load_config(path).raw.get("subset_fraction", 1.0) == 1.0, (
            f"{path.name}: subset_fraction < 1.0 - debug runs must not be committed"
        )


def test_every_stage_config_declares_its_io(stage_configs: list) -> None:
    """The seam must be readable from the config alone, without reading code."""
    import yaml

    assert stage_configs, "no stage configs found"
    for path in stage_configs:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "stage" in data, f"{path.name}: missing 'stage'"
        assert "owner" in data, f"{path.name}: missing 'owner' - who do I ask about this?"
        assert "outputs" in data, f"{path.name}: missing 'outputs' - what does it hand on?"
        assert "contract" in data, f"{path.name}: missing 'contract'"


def test_run_id_mismatch_is_rejected(tmp_path) -> None:
    """A declared run_id that disagrees with model/optimizer/seed must fail loudly."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "extends: [models, training]\n"
        "run_id: something__else__seed42\n"
        "model:\n  name: model_b\n"
        "optimizer:\n  name: adam\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="run_id"):
        load_config(bad)
