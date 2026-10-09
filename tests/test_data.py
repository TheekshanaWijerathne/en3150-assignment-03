"""Unit tests for the data layer: split drawing, manifests, pixel cache, synthetic data,
loaders, and EuroSAT preparation (on a small fake download) with its Section 1 figures."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from edgecnn.config.loader import load_config
from edgecnn.contracts import paths, schema
from edgecnn.contracts.schema import ContractViolation
from edgecnn.contracts.types import INPUT_SHAPE, SPLIT_FRACTIONS
from edgecnn.data.prepare import (
    ensure_images_present,
    prepare_dataset,
    read_split_manifest,
    stratified_split,
    summarize_split,
    write_split_manifest,
)
from edgecnn.utils.io import sha256_file

torch = pytest.importorskip("torch", reason="torch not installed")

from edgecnn.data.cache import channel_stats, load_pixels  # noqa: E402
from edgecnn.data.inspect import (  # noqa: E402
    plot_class_distribution,
    plot_sample_grid,
    split_counts_table,
)
from edgecnn.data.loaders import (  # noqa: E402
    build_dataloaders,
    build_single_loader,
    compute_norm_stats,
)
from edgecnn.data.synthetic import (  # noqa: E402
    EUROSAT_CLASSES,
    fixture_is_current,
    make_synthetic_fixture,
)
from edgecnn.data.transforms import build_transform  # noqa: E402

FRACTIONS = dict(SPLIT_FRACTIONS)
EUROSAT_CLASS_SIZES = [3000, 3000, 3000, 2500, 2500, 2000, 2500, 3000, 2500, 3000]
MEAN, STD = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)


def _experiment(name: str) -> str:
    return str(paths.CONFIGS_DIR / "experiments" / f"{name}.yaml")


# --- drawing the split -----------------------------------------------------------


def test_split_gives_exact_eurosat_counts() -> None:
    labels = [label for label, size in enumerate(EUROSAT_CLASS_SIZES) for _ in range(size)]
    splits = stratified_split(labels, FRACTIONS, seed=42)
    assert {name: splits.count(name) for name in FRACTIONS} == {
        "train": 18900, "val": 4050, "test": 4050,
    }


def test_split_keeps_every_class_in_proportion() -> None:
    labels = [label for label, size in enumerate(EUROSAT_CLASS_SIZES) for _ in range(size)]
    splits = stratified_split(labels, FRACTIONS, seed=42)
    for label, size in enumerate(EUROSAT_CLASS_SIZES):
        own = [split for lab, split in zip(labels, splits, strict=True) if lab == label]
        assert own.count("train") == round(0.70 * size)
        assert own.count("val") == round(0.15 * size)


def test_split_is_reproducible_and_depends_on_the_seed() -> None:
    labels = [index % 5 for index in range(500)]
    assert stratified_split(labels, FRACTIONS, 1) == stratified_split(labels, FRACTIONS, 1)
    assert stratified_split(labels, FRACTIONS, 1) != stratified_split(labels, FRACTIONS, 2)


def test_split_never_leaves_a_split_empty_for_small_classes() -> None:
    labels = [0] * 4 + [1] * 5
    splits = stratified_split(labels, FRACTIONS, 0)
    for label in (0, 1):
        own = {split for lab, split in zip(labels, splits, strict=True) if lab == label}
        assert own == {"train", "val", "test"}


def test_split_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError):
        stratified_split([0, 1], {"train": 0.8, "test": 0.2}, 0)
    with pytest.raises(ValueError):
        stratified_split([0, 1], {"train": 0.8, "val": 0.15, "test": 0.15}, 0)


def test_manifest_is_identical_on_every_platform(tmp_path) -> None:
    row = {"relative_path": "A\\x.png", "label_index": 0, "label_name": "A", "split": "train"}
    path = write_split_manifest(tmp_path / "manifest.csv", [row])
    raw = path.read_bytes()
    assert b"\r\n" not in raw  # Windows line endings would change the SHA-256
    assert b"A/x.png" in raw
    assert read_split_manifest(path)[0]["label_index"] == 0


def test_summarize_split_counts_per_class() -> None:
    rows = [
        {"split": "train", "label_name": "A"},
        {"split": "train", "label_name": "A"},
        {"split": "test", "label_name": "B"},
    ]
    counts, per_class = summarize_split(rows)
    assert counts == {"train": 2, "val": 0, "test": 1, "total": 3}
    assert per_class["train"] == {"A": 2}


# --- pixel cache -----------------------------------------------------------------


def test_channel_stats_match_numpy() -> None:
    pixels = np.random.RandomState(0).randint(0, 256, size=(10, 8, 8, 3), dtype=np.uint8)
    mean, std = channel_stats(pixels, [0, 1, 2, 3, 4], chunk=2)
    reference = pixels[:5].astype(np.float64) / 255
    assert np.allclose(mean, reference.mean(axis=(0, 1, 2)))
    assert np.allclose(std, reference.std(axis=(0, 1, 2)))


def test_pixel_cache_is_reused_then_rebuilt_when_an_image_changes(tmp_path) -> None:
    from PIL import Image

    root = make_synthetic_fixture(tmp_path, num_classes=2, images_per_class=5)
    rels = [row["relative_path"] for row in read_split_manifest(root / "split_manifest.csv")]
    first = sorted(root.glob("images.pixels-*.npy"))
    assert len(first) == 1

    load_pixels(root / "images", rels)
    assert sorted(root.glob("images.pixels-*.npy")) == first  # reused, not rebuilt

    changed = Image.open(root / "images" / rels[0]).copy()
    changed.putpixel((0, 0), (0, 0, 0))
    changed.save(root / "images" / rels[0])
    pixels = load_pixels(root / "images", rels)
    second = sorted(root.glob("images.pixels-*.npy"))
    assert len(second) == 1 and second != first  # a new cache replaced the stale one
    assert tuple(pixels[0, 0, 0]) == (0, 0, 0)


def test_pixel_cache_reports_missing_images(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        load_pixels(tmp_path, ["nope.png"])


# --- synthetic data --------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    return make_synthetic_fixture(tmp_path_factory.mktemp("synthetic"))


def test_fixture_has_ten_eurosat_classes_and_valid_files(fixture_root) -> None:
    meta = schema.read_json(fixture_root / "split_meta.json", "split_meta")
    assert meta["class_names"] == list(EUROSAT_CLASSES)
    assert meta["counts"] == {"train": 140, "val": 30, "test": 30, "total": 200}
    schema.read_json(fixture_root / "norm_stats.json", "norm_stats")
    assert fixture_is_current(fixture_root)


def test_fixture_is_deterministic(tmp_path, fixture_root) -> None:
    again = make_synthetic_fixture(tmp_path / "again")
    manifest = "split_manifest.csv"
    assert (again / manifest).read_bytes() == (fixture_root / manifest).read_bytes()
    rels = [row["relative_path"] for row in read_split_manifest(again / manifest)]
    pixels = [load_pixels(root / "images", rels) for root in (again, fixture_root)]
    assert np.array_equal(*pixels)


def test_fixture_regeneration_touches_only_its_own_files(tmp_path) -> None:
    keep = tmp_path / "notes.txt"
    keep.write_text("unrelated", encoding="utf-8")
    make_synthetic_fixture(tmp_path, num_classes=2, images_per_class=5)
    make_synthetic_fixture(tmp_path, num_classes=2, images_per_class=5)
    assert keep.read_text(encoding="utf-8") == "unrelated"


# --- transforms ------------------------------------------------------------------


def test_transform_output_matches_the_batch_contract() -> None:
    transform = build_transform("val", MEAN, STD)
    out = transform(torch.full((3, 64, 64), 128, dtype=torch.uint8))
    assert out.dtype == torch.float32 and out.shape == (3, 64, 64)
    # 128/255 - 0.5 cancels most float32 digits, so compare with an absolute tolerance
    assert torch.allclose(out, torch.full_like(out, (128 / 255 - 0.5) / 0.25), rtol=0, atol=1e-6)


def test_validation_and_test_are_never_augmented() -> None:
    with pytest.warns(UserWarning, match="train only"):
        transform = build_transform("test", MEAN, STD, ["random_horizontal_flip"])
    image = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    assert torch.equal(transform(image), transform(image))


def test_training_augmentation_varies_between_calls() -> None:
    torch.manual_seed(0)
    transform = build_transform(
        "train", MEAN, STD, ["random_horizontal_flip", "random_vertical_flip", "random_rotation_90"]
    )
    image = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    outputs = [transform(image) for _ in range(20)]
    assert any(not torch.equal(outputs[0], other) for other in outputs[1:])


def test_unknown_augmentation_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown augmentation"):
        build_transform("train", MEAN, STD, ["random_hozirontal_flip"])


def test_resize_is_added_only_when_a_model_needs_another_size() -> None:
    image = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    upsized = build_transform("val", MEAN, STD, image_size=(128, 128), source_size=(64, 64))
    native = build_transform("val", MEAN, STD, image_size=(64, 64), source_size=(64, 64))
    assert upsized(image).shape == (3, 128, 128)
    assert native(image).shape == (3, 64, 64)


# --- loaders (synthetic mode, isolated from the repo's fixture folder) ------------


@pytest.fixture
def use_fixture(monkeypatch, fixture_root):
    monkeypatch.setattr(paths, "SYNTHETIC_FIXTURE_DIR", fixture_root)
    return fixture_root


def test_bundle_satisfies_the_data_contract(use_fixture) -> None:
    data = build_dataloaders(load_config(_experiment("model_b__adam"), mode="synthetic"))
    images, labels = next(iter(data.train))
    assert images.dtype == torch.float32 and images.shape[1:] == INPUT_SHAPE
    assert labels.dtype == torch.int64
    assert data.num_classes == 10 and data.class_names == list(EUROSAT_CLASSES)
    assert (len(data.train.dataset), len(data.val.dataset), len(data.test.dataset)) == (140, 30, 30)


def test_training_statistics_normalise_the_training_split(use_fixture) -> None:
    data = build_dataloaders(load_config(_experiment("model_b__adam"), mode="synthetic"))
    images = torch.cat([batch for batch, _ in data.train])
    assert torch.allclose(images.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-3)
    assert torch.allclose(images.std(dim=(0, 2, 3)), torch.ones(3), atol=1e-2)


def test_validation_order_is_fixed(use_fixture) -> None:
    data = build_dataloaders(load_config(_experiment("model_b__adam"), mode="synthetic"))
    first = list(data.val)
    second = list(data.val)
    for (images_a, labels_a), (images_b, labels_b) in zip(first, second, strict=True):
        assert torch.equal(labels_a, labels_b) and torch.equal(images_a, images_b)


def test_training_order_is_reproducible(use_fixture) -> None:
    cfg = load_config(_experiment("model_b__adam"), mode="synthetic")
    first = next(iter(build_dataloaders(cfg).train))[1]
    second = next(iter(build_dataloaders(cfg).train))[1]
    assert torch.equal(first, second)


def test_subset_fraction_shrinks_training_only(use_fixture) -> None:
    cfg = load_config(_experiment("model_b__adam"), mode="synthetic", subset_fraction=0.5)
    data = build_dataloaders(cfg)
    assert len(data.train.dataset) == 70 and len(data.val.dataset) == 30
    assert set(data.train.dataset.labels.tolist()) == set(range(10))  # every class kept


def test_pretrained_backbones_use_imagenet_statistics(use_fixture) -> None:
    data = build_dataloaders(load_config(_experiment("mobilenet_v2__adam"), mode="synthetic"))
    assert data.norm_mean == (0.485, 0.456, 0.406)


def test_custom_models_use_training_statistics(use_fixture) -> None:
    cfg = load_config(_experiment("model_b__adam"), mode="synthetic")
    stats = schema.read_json(use_fixture / "norm_stats.json", "norm_stats")
    assert list(build_dataloaders(cfg).norm_mean) == stats["mean"]
    assert np.allclose(compute_norm_stats(cfg)["mean"], stats["mean"])


def test_resolution_ablation_resizes_every_split(use_fixture) -> None:
    cfg = load_config(
        _experiment("mobilenet_v2__adam"), mode="synthetic", **{"pretrained.input_resolution": 128}
    )
    data = build_dataloaders(cfg)
    assert data.input_shape == (3, 128, 128)
    assert next(iter(data.val))[0].shape[1:] == (3, 128, 128)


def test_single_loader_and_bad_split_names(use_fixture) -> None:
    cfg = load_config(_experiment("model_b__adam"), mode="synthetic")
    assert len(build_single_loader(cfg, "test").dataset) == 30
    with pytest.raises(ValueError):
        build_single_loader(cfg, "holdout")


def test_real_data_without_a_committed_split_says_what_to_do(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(paths, "SPLIT_MANIFEST", tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError, match="01_data_preparation"):
        build_dataloaders(load_config(_experiment("model_b__adam"), mode="debug"))


# --- EuroSAT preparation, on a small fake download (fake_eurosat: tests/conftest.py) ---

def test_debug_run_draws_the_split_but_writes_nothing(fake_eurosat) -> None:
    prepared = prepare_dataset(fake_eurosat.config("debug"))
    assert prepared["written"] == [] and not paths.SPLIT_MANIFEST.exists()
    assert prepared["meta"]["class_names"] == list(fake_eurosat.classes)
    assert prepared["meta"]["counts"] == {"train": 42, "val": 9, "test": 9, "total": 60}
    assert prepared["norm_stats"]["fitted_on"] == "train"
    assert prepared["norm_stats"]["num_images"] == 42
    assert len(list(fake_eurosat.images.rglob("*.jpg"))) == 60


def test_official_run_writes_the_split_debug_showed(fake_eurosat) -> None:
    shown = prepare_dataset(fake_eurosat.config("debug"))["meta"]["manifest_sha256"]
    written = prepare_dataset(fake_eurosat.config("official"))["written"]
    assert set(written) == {paths.SPLIT_MANIFEST, paths.SPLIT_META, paths.NORM_STATS}
    assert schema.validate_manifest(paths.SPLIT_MANIFEST) == 60
    meta = schema.read_json(paths.SPLIT_META, "split_meta")
    assert meta["manifest_sha256"] == sha256_file(paths.SPLIT_MANIFEST) == shown
    schema.read_json(paths.NORM_STATS, "norm_stats")


def test_processed_images_are_byte_copies(fake_eurosat) -> None:
    prepare_dataset(fake_eurosat.config("debug"))
    source = fake_eurosat.root / "raw" / "eurosat" / "2750" / "River" / "River_7.jpg"
    assert (fake_eurosat.images / "River" / "River_7.jpg").read_bytes() == source.read_bytes()


def test_a_committed_split_is_reused_never_redrawn(fake_eurosat) -> None:
    prepare_dataset(fake_eurosat.config("official"))
    committed = paths.SPLIT_MANIFEST.read_bytes()
    reseeded = fake_eurosat.config("official", **{"split.seed": 7})
    assert prepare_dataset(reseeded)["written"] == []
    assert paths.SPLIT_MANIFEST.read_bytes() == committed
    assert prepare_dataset(reseeded, force=True)["written"]
    assert paths.SPLIT_MANIFEST.read_bytes() != committed


def test_an_edited_manifest_is_refused(fake_eurosat) -> None:
    prepare_dataset(fake_eurosat.config("official"))
    edited = paths.SPLIT_MANIFEST.read_text(encoding="utf-8").replace(",train", ",test", 1)
    paths.SPLIT_MANIFEST.write_text(edited, encoding="utf-8", newline="\n")
    with pytest.raises(ContractViolation, match="SHA-256"):
        prepare_dataset(fake_eurosat.config("official"))
    with pytest.raises(ContractViolation, match="SHA-256"):
        ensure_images_present(fake_eurosat.config("debug"))


def test_an_incomplete_download_is_refused(fake_eurosat) -> None:
    with pytest.raises(ValueError, match="expected_num_images"):
        prepare_dataset(fake_eurosat.config("debug", **{"dataset.expected_num_images": 61}))


def test_a_missing_download_explains_what_to_do(fake_eurosat) -> None:
    nowhere = fake_eurosat.config("debug", **{"dataset.root": str(fake_eurosat.root / "nowhere")})
    with pytest.raises(FileNotFoundError, match="2750"):
        prepare_dataset(nowhere)


def test_missing_images_are_fetched_without_redrawing(fake_eurosat) -> None:
    prepare_dataset(fake_eurosat.config("official"))
    committed = paths.SPLIT_MANIFEST.read_bytes()
    for name in ("Forest/Forest_1.jpg", "SeaLake/SeaLake_20.jpg"):
        (fake_eurosat.images / name).unlink()
    cfg = fake_eurosat.config("debug")
    assert ensure_images_present(cfg) == 2
    assert ensure_images_present(cfg) == 0
    assert paths.SPLIT_MANIFEST.read_bytes() == committed


def test_a_fresh_clone_loads_the_committed_split(fake_eurosat) -> None:
    prepare_dataset(fake_eurosat.config("official"))
    shutil.rmtree(paths.PROCESSED_DIR)  # images and pixel cache gone, as on a new machine
    data = build_dataloaders(fake_eurosat.config("debug"))
    assert data.class_names == list(fake_eurosat.classes)
    assert (len(data.val.dataset), len(data.test.dataset)) == (9, 9)
    images, labels = next(iter(data.test))
    assert images.dtype == torch.float32 and images.shape[1:] == INPUT_SHAPE
    assert labels.dtype == torch.int64


def test_split_table_and_figures(fake_eurosat) -> None:
    meta = prepare_dataset(fake_eurosat.config("debug"))["meta"]
    table = split_counts_table(meta)
    assert table.loc["Total"].tolist() == [42, 9, 9, 60]
    assert table.loc["River"].tolist() == [14, 3, 3, 20]

    plot_class_distribution(meta, write=True)
    assert (paths.DATASET_FIGURES_DIR / "class_distribution.png").exists()

    cfg = fake_eurosat.config("debug")
    grids = [plot_sample_grid(cfg, per_class=2, write=True) for _ in range(2)]
    shown = [[ax.images[0].get_array() for ax in grid.axes if ax.images] for grid in grids]
    assert len(shown[0]) == len(fake_eurosat.classes) * 2
    assert all(np.array_equal(a, b) for a, b in zip(*shown, strict=True))  # same picks every time
    assert (paths.DATASET_FIGURES_DIR / "sample_grid.png").exists()


def test_synthetic_mode_prepares_the_generated_split(use_fixture) -> None:
    cfg = load_config(_experiment("model_b__adam"), mode="synthetic")
    prepared = prepare_dataset(cfg)
    assert prepared["written"] == [] and prepared["meta"]["dataset_name"] == "synthetic"
    assert prepared["meta"]["counts"]["total"] == 200
    assert sum(bool(ax.images) for ax in plot_sample_grid(cfg, per_class=2).axes) == 20


def test_split_files_keep_lf_line_endings_in_git() -> None:
    rules = (paths.REPO_ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "data/splits/* text eol=lf" in rules
