"""Draw the train / validation / test split and write the files that define it.

The split is drawn once and saved as three files: the manifest (which image is
in which split), its metadata, and normalisation statistics computed from the
training images only. Every model then trains and is tested on byte-identical
splits, so an accuracy difference between two models comes from the models,
not from the data.

Team notes:
Owner: Member 1. The shared helpers at the bottom are also used by the
synthetic fixture.

    IN   cfg.section("dataset"), cfg.section("split")
    OUT  data/raw/eurosat/                git-ignored: the download
         data/processed/eurosat_64/       git-ignored, regenerable
         data/splits/split_manifest.csv   COMMITTED } official mode only
         data/splits/split_meta.json      COMMITTED }
         data/splits/norm_stats.json      COMMITTED }
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from edgecnn.contracts import paths, schema
from edgecnn.contracts.schema import MANIFEST_COLUMNS, ContractViolation
from edgecnn.contracts.types import SPLIT_NAMES
from edgecnn.utils.io import repo_relative, sha256_file
from edgecnn.utils.logging import get_logger, progress

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from edgecnn.contracts.types import ResolvedConfig

log = get_logger(__name__)

#: How images are brought to the final resolution (``dataset.resize_mode``).
#: EuroSAT is natively 64x64, so it uses ``none``: the files are copied as they are.
RESIZE_MODES: tuple[str, ...] = ("none", "bilinear")


@dataclass(frozen=True)
class _Prepared:
    """One prepared dataset: its manifest rows, metadata, statistics and image folder."""

    rows: list[dict[str, Any]]
    meta: dict[str, Any]
    stats: dict[str, Any]
    written: list[Path]
    image_dir: Path


def processed_image_dir(cfg: ResolvedConfig) -> Path:
    """Folder holding a dataset's images at the final resolution.

    ``data/processed/<dataset>_<height>``, e.g. ``data/processed/eurosat_64``.
    Every ``relative_path`` in the manifest is relative to this folder.
    """
    name = cfg.section("dataset").get("name", "eurosat")
    height = int(cfg.raw.get("image_size", [64, 64])[0])
    return paths.PROCESSED_DIR / f"{name}_{height}"


def prepare_dataset(cfg: ResolvedConfig, force: bool = False) -> dict[str, Any]:
    """Prepare the dataset and its split. Safe to re-run.

    1. Downloads EuroSAT through torchvision into ``dataset.root`` (once) and
       brings every image into :func:`processed_image_dir` at the final size.
    2. Draws the stratified split from ``split.seed``. If a split is already
       committed, it is checked against the images and reused instead - it is
       only redrawn with ``force=True``.
    3. Computes the normalisation statistics from the training images only.
    4. Writes the three split files only if ``cfg.is_official``. Every mode
       returns everything, so the split can be inspected before it is written.

    With ``dataset.name == "synthetic"`` it returns the generated dataset's
    split instead, and never writes.

    Args:
        force: Redraw and overwrite an existing committed split. Doing so
            invalidates every result already in ``results/metrics/``.

    Returns:
        ``{"manifest": rows, "meta": dict, "norm_stats": dict, "written": [paths]}``.

    Raises:
        FileNotFoundError: if the dataset can't be downloaded or found.
        ValueError: if the images don't match ``dataset.expected_num_classes``
            and ``expected_num_images`` - a partial download must never become
            the split.
        ContractViolation: if the committed split was edited after it was
            written, or was drawn from different images.
    """
    prepared = _prepare(cfg, force=force, write=cfg.is_official)
    return {
        "manifest": prepared.rows,
        "meta": prepared.meta,
        "norm_stats": prepared.stats,
        "written": prepared.written,
    }


def ensure_images_present(cfg: ResolvedConfig) -> int:
    """Make every image listed in the committed manifest exist locally.

    The manifest is committed; the images are not. A fresh clone - a
    teammate's laptop, a Colab session - therefore has the split but no
    pictures. This downloads the dataset and prepares exactly the missing
    images. It never redraws the split: it first checks that the manifest is
    still the file whose SHA-256 ``split_meta.json`` recorded.

    Returns:
        The number of images that had to be fetched (0 when all were present).

    Raises:
        ContractViolation: if the manifest was edited after it was committed,
            or lists images that the downloaded dataset does not contain.
    """
    if cfg.section("dataset").get("name") == "synthetic":
        from edgecnn.data.synthetic import make_synthetic_fixture

        make_synthetic_fixture(paths.SYNTHETIC_FIXTURE_DIR)
        return len(read_split_manifest(paths.SYNTHETIC_FIXTURE_DIR / "split_manifest.csv"))

    meta = schema.read_json(paths.SPLIT_META, "split_meta")
    _check_manifest_hash(paths.SPLIT_MANIFEST, meta)
    image_dir = processed_image_dir(cfg)
    rows = read_split_manifest(paths.SPLIT_MANIFEST)
    missing = [
        row["relative_path"] for row in rows if not (image_dir / row["relative_path"]).exists()
    ]
    if not missing:
        return 0

    log.info("%d of %d images are missing locally - fetching them", len(missing), len(rows))
    resize_mode = _resize_mode(cfg)
    _, sources = _raw_images(cfg)
    by_path = {_processed_path(source, name, resize_mode): source for source, name in sources}
    unknown = [rel for rel in missing if rel not in by_path]
    if unknown:
        raise ContractViolation(
            f"{len(unknown)} image(s) in the committed split are not in the downloaded dataset, "
            f"e.g. {unknown[0]!r}: the split was drawn from a different version of it"
        )
    _materialise([(by_path[rel], rel) for rel in missing], image_dir, resize_mode, _image_size(cfg))
    return len(missing)


# --- preparing a dataset ------------------------------------------------------


def _prepare(
    cfg: ResolvedConfig, *, force: bool, write: bool, with_stats: bool = True
) -> _Prepared:
    """Prepare ``cfg``'s dataset and split; write the split files only if ``write``.

    ``with_stats=False`` skips computing the normalisation statistics of a new
    split (the returned ``stats`` is then empty) - for callers that only need
    the rows, such as the sample grid.
    """
    if cfg.section("dataset").get("name") == "synthetic":
        return _prepare_synthetic()

    image_dir = processed_image_dir(cfg)
    classes, rows = _collect(cfg, image_dir)
    if paths.SPLIT_MANIFEST.exists():
        if not force:
            return _committed(rows, image_dir)
        if write:
            log.warning("force=True: overwriting the committed split; results from it are stale")

    split = cfg.section("split")
    fractions = {name: float(split[name]) for name in SPLIT_NAMES}
    seed = int(split.get("seed", 42))
    stratified = bool(split.get("stratified", True))
    groups = [row["label_index"] for row in rows] if stratified else [0] * len(rows)
    for row, name in zip(rows, stratified_split(groups, fractions, seed), strict=True):
        row["split"] = name

    height, width = _image_size(cfg)
    stats: dict[str, Any] = {}
    if with_stats or write:
        from edgecnn.data.cache import channel_stats, load_pixels

        pixels = load_pixels(image_dir, [row["relative_path"] for row in rows], (height, width))
        train = [position for position, row in enumerate(rows) if row["split"] == "train"]
        stats = _norm_stats_payload(*channel_stats(pixels, train), len(train))
        schema.validate(stats, "norm_stats")
    counts, per_class = summarize_split(rows)
    meta = {
        "dataset_name": str(cfg.section("dataset")["name"]),
        "dataset_source": _source_note(cfg.section("dataset")),
        "num_classes": len(classes),
        "class_names": classes,
        "image_size": [height, width],
        "seed": seed,
        "stratified": stratified,
        "fractions": fractions,
        "counts": counts,
        "per_class_counts": per_class,
        # the hash of the file an official run writes, so every mode shows the same one
        "manifest_sha256": hashlib.sha256(_manifest_bytes(rows)).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    schema.validate(meta, "split_meta")

    written: list[Path] = []
    if write:
        fields = {key: value for key, value in meta.items() if key != "manifest_sha256"}
        written = [
            write_split_manifest(paths.SPLIT_MANIFEST, rows),
            write_norm_stats(paths.NORM_STATS, stats["mean"], stats["std"], stats["num_images"]),
            write_split_meta(paths.SPLIT_META, paths.SPLIT_MANIFEST, **fields),
        ]
        log.info("Wrote the split: %s", ", ".join(repo_relative(path) for path in written))
    return _Prepared(rows, meta, stats, written, image_dir)


def _prepare_synthetic() -> _Prepared:
    """The generated dataset's split, generating it first if needed. Never writes."""
    from edgecnn.data.synthetic import IMAGE_DIR_NAME, fixture_is_current, make_synthetic_fixture

    root = paths.SYNTHETIC_FIXTURE_DIR
    if not fixture_is_current(root):
        make_synthetic_fixture(root)
    return _Prepared(
        rows=read_split_manifest(root / "split_manifest.csv"),
        meta=schema.read_json(root / "split_meta.json", "split_meta"),
        stats=schema.read_json(root / "norm_stats.json", "norm_stats"),
        written=[],
        image_dir=root / IMAGE_DIR_NAME,
    )


def _committed(rows: list[dict[str, Any]], image_dir: Path) -> _Prepared:
    """The committed split, after checking it against its hash and the prepared images."""
    meta = schema.read_json(paths.SPLIT_META, "split_meta")
    _check_manifest_hash(paths.SPLIT_MANIFEST, meta)
    committed = read_split_manifest(paths.SPLIT_MANIFEST)
    listed = {(row["relative_path"], row["label_name"]) for row in committed}
    found = {(row["relative_path"], row["label_name"]) for row in rows}
    if listed != found:
        raise ContractViolation(
            f"the committed split and the prepared images differ ({len(listed - found)} listed "
            f"images are missing, {len(found - listed)} images are not listed): it was drawn from "
            "a different version of the dataset"
        )
    log.info("Using the committed split %s (force=True redraws it)", meta["manifest_sha256"][:12])
    stats = schema.read_json(paths.NORM_STATS, "norm_stats")
    return _Prepared(committed, meta, stats, [], image_dir)


def _collect(cfg: ResolvedConfig, image_dir: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Fetch the raw dataset, prepare it into ``image_dir`` and list it.

    Returns:
        ``(class_names, rows)``: classes in label order (sorted folder names),
        and one row per image, sorted by ``relative_path``, without a split yet.
    """
    classes, sources = _raw_images(cfg)
    dataset = cfg.section("dataset")
    checks = (("expected_num_classes", len(classes)), ("expected_num_images", len(sources)))
    for key, found in checks:
        expected = dataset.get(key)
        if expected is not None and int(expected) != found:
            raise ValueError(
                f"found {found} but dataset.{key} is {expected}: the download is incomplete or a "
                f"different version. Delete {repo_relative(_raw_root(cfg))} and run again."
            )
    resize_mode = _resize_mode(cfg)
    jobs = [(source, _processed_path(source, name, resize_mode)) for source, name in sources]
    _materialise(jobs, image_dir, resize_mode, _image_size(cfg))

    label = {name: index for index, name in enumerate(classes)}
    rows = [
        {"relative_path": rel, "label_index": label[name], "label_name": name}
        for (_, name), (_, rel) in zip(sources, jobs, strict=True)
    ]
    rows.sort(key=lambda row: row["relative_path"])
    return classes, rows


def _raw_images(cfg: ResolvedConfig) -> tuple[list[str], list[tuple[Path, str]]]:
    """The raw dataset's classes and ``(image, class name)`` pairs, downloading it if allowed."""
    import torchvision.datasets

    dataset = cfg.section("dataset")
    if dataset.get("source", "torchvision") != "torchvision":
        raise ValueError(
            f"dataset.source {dataset.get('source')!r} is not supported; use 'torchvision'"
        )
    name = dataset.get("torchvision_class", "EuroSAT")
    root = _raw_root(cfg)
    download = bool(dataset.get("download", True))
    try:
        raw = getattr(torchvision.datasets, name)(root=str(root), download=download)
    except Exception as exc:  # no network, a corrupt archive, or not downloaded yet
        reason = "the download failed" if download else "it is missing and dataset.download is off"
        raise FileNotFoundError(
            f"could not load {name} from {repo_relative(root)}: {reason} ({exc}). To add it by "
            f"hand, unzip EuroSAT.zip so that {repo_relative(root / 'eurosat' / '2750')} holds one "
            "folder per class."
        ) from exc
    classes = list(raw.classes)
    return classes, [(Path(path), classes[target]) for path, target in raw.samples]


def _materialise(
    jobs: Sequence[tuple[Path, str]],
    image_dir: Path,
    resize_mode: str,
    image_size: tuple[int, int],
) -> None:
    """Bring each ``(source image, relative path)`` into ``image_dir``, skipping finished ones.

    With ``resize_mode="none"`` the file is copied byte for byte, so the
    processed image is exactly the downloaded one. An interrupted run is simply
    resumed: a copy whose size differs from its source is made again.
    """
    height, width = image_size
    listings: dict[Path, dict[str, int]] = {}

    def sizes(folder: Path) -> dict[str, int]:
        """File name -> size for one folder, from a single directory listing."""
        if folder not in listings:
            try:
                with os.scandir(folder) as entries:
                    listings[folder] = {e.name: e.stat().st_size for e in entries if e.is_file()}
            except FileNotFoundError:
                listings[folder] = {}
        return listings[folder]

    def done(job: tuple[Path, str]) -> bool:
        source, rel = job
        target = image_dir / rel
        have = sizes(target.parent).get(target.name)
        if resize_mode == "none":
            return have is not None and have == sizes(source.parent).get(source.name)
        return have is not None

    def one(job: tuple[Path, str]) -> None:
        source, rel = job
        target = image_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if resize_mode == "none":
            shutil.copyfile(source, target)
            return
        from PIL import Image

        with Image.open(source) as image:
            image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR).save(target)

    todo = [job for job in jobs if not done(job)]
    if not todo:  # the usual case after the first run: nothing to do, and no progress bar
        return
    workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in progress(pool.map(one, todo), len(todo), "Preparing images", "img"):
            pass


def _processed_path(source: Path, class_name: str, resize_mode: str) -> str:
    """Where a raw image goes, relative to the processed folder: ``<class>/<file>``."""
    name = source.name if resize_mode == "none" else f"{source.stem}.png"  # resized: lossless
    return f"{class_name}/{name}"


def _check_manifest_hash(manifest: Path, meta: dict[str, Any]) -> None:
    """Raise unless ``manifest`` is the file whose SHA-256 ``split_meta.json`` recorded."""
    if sha256_file(manifest) != meta["manifest_sha256"]:
        raise ContractViolation(
            f"{repo_relative(manifest)} no longer matches the SHA-256 recorded in split_meta.json: "
            "it was edited or regenerated after the split was committed (or its line endings were "
            "converted), so results drawn from the committed split may not match it. Restore it "
            "with `git checkout -- data/splits/`."
        )


def _resize_mode(cfg: ResolvedConfig) -> str:
    mode = str(cfg.section("dataset").get("resize_mode", "none"))
    if mode not in RESIZE_MODES:
        raise ValueError(f"dataset.resize_mode must be one of {list(RESIZE_MODES)}, not {mode!r}")
    return mode


def _image_size(cfg: ResolvedConfig) -> tuple[int, int]:
    height, width = cfg.raw.get("image_size", [64, 64])
    return int(height), int(width)


def _raw_root(cfg: ResolvedConfig) -> Path:
    from edgecnn.config.loader import resolve_path

    return resolve_path(cfg.section("dataset").get("root", "data/raw"))


def _source_note(dataset: dict[str, Any]) -> str:
    """Where the images came from, for the report's dataset citation."""
    note = f"torchvision.datasets.{dataset.get('torchvision_class', 'EuroSAT')}"
    citation = dataset.get("citation")
    return f"{note}; {citation}" if citation else note


# --- shared helpers (also used by the synthetic fixture) ----------------------


def stratified_split(
    labels: Sequence[int],
    fractions: dict[str, float],
    seed: int,
) -> list[str]:
    """Assign every sample to ``train``, ``val`` or ``test``, class by class.

    Within each class the samples are shuffled, then cut in the given
    proportions (each count rounded to the nearest whole image). Splitting per
    class keeps the class balance identical in all three splits, and guarantees
    every class appears in each of them whenever it has at least three samples.

    The shuffle uses NumPy's legacy ``RandomState``, whose sequence of random
    numbers is frozen across NumPy versions, so the same seed reproduces the
    same split on any machine.

    Args:
        labels: The integer class of each sample, in manifest order.
        fractions: ``{"train": 0.70, "val": 0.15, "test": 0.15}``.
        seed: Random seed.

    Returns:
        The split name of each sample, in the same order as ``labels``.

    Raises:
        ValueError: if the fractions don't name exactly train / val / test, or
            don't sum to 1.
    """
    import numpy as np

    if set(fractions) != set(SPLIT_NAMES):
        raise ValueError(
            f"fractions must name exactly {list(SPLIT_NAMES)}, got {sorted(fractions)}"
        )
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {sum(fractions.values())}")

    labels = np.asarray(labels)
    assignment = np.empty(len(labels), dtype=object)
    rng = np.random.RandomState(seed)
    for label in np.unique(labels):  # sorted, so the random sequence is fixed
        members = rng.permutation(np.flatnonzero(labels == label))
        start = 0
        for name, count in zip(SPLIT_NAMES, _split_sizes(len(members), fractions), strict=True):
            assignment[members[start : start + count]] = name
            start += count
    return assignment.tolist()


def _split_sizes(n: int, fractions: dict[str, float]) -> list[int]:
    """Train / val / test sizes for one class of ``n`` samples."""
    sizes = [round(n * fractions["train"]), round(n * fractions["val"])]
    sizes.append(n - sum(sizes))
    if n >= 3:  # move samples from train so no split is empty
        for index in (1, 2):
            if sizes[index] < 1:
                sizes[0] -= 1 - sizes[index]
                sizes[index] = 1
    return sizes


def write_split_manifest(manifest_path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write ``split_manifest.csv``: one row per image, columns in the fixed order.

    Columns: ``relative_path, label_index, label_name, split``. Paths use forward
    slashes and lines end in ``\\n`` on every platform, so the file - and its
    SHA-256 recorded in the metadata - is identical on Windows, Linux and Colab.
    The written file is validated before returning.

    Raises:
        ContractViolation: if any row breaks the manifest schema.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(_manifest_bytes(rows))
    schema.validate_manifest(manifest_path)
    return manifest_path


def _manifest_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    """The exact bytes of ``split_manifest.csv`` for ``rows``."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(MANIFEST_COLUMNS)
    for row in rows:
        writer.writerow([
            str(row["relative_path"]).replace("\\", "/"),
            int(row["label_index"]),
            row["label_name"],
            row["split"],
        ])
    return buffer.getvalue().encode("utf-8")


def read_split_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Read and validate ``split_manifest.csv``; ``label_index`` comes back as ``int``.

    Raises:
        FileNotFoundError: if the manifest does not exist.
        ContractViolation: if it breaks the manifest schema.
    """
    manifest_path = Path(manifest_path)
    schema.validate_manifest(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return [
            {**row, "label_index": int(row["label_index"])}
            for row in csv.DictReader(handle)
        ]


def summarize_split(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Image counts per split, and per class within each split.

    Returns:
        ``({"train": n, "val": n, "test": n, "total": n},
        {"train": {class_name: n, ...}, ...})``.
    """
    counts = {name: 0 for name in SPLIT_NAMES}
    per_class: dict[str, dict[str, int]] = {name: {} for name in SPLIT_NAMES}
    for row in rows:
        counts[row["split"]] += 1
        classes = per_class[row["split"]]
        classes[row["label_name"]] = classes.get(row["label_name"], 0) + 1
    counts["total"] = sum(counts[name] for name in SPLIT_NAMES)
    return counts, per_class


def write_split_meta(meta_path: Path, manifest_path: Path, **fields: Any) -> Path:
    """Write ``split_meta.json``, including the manifest's SHA-256.

    The hash lets anyone confirm that the manifest on disk is the one the
    reported results were produced with: if it stops matching, the split was
    regenerated or edited. ``created_utc`` is filled in unless given.

    Raises:
        ContractViolation: if the metadata breaks its schema.
    """
    payload = dict(fields)
    payload["manifest_sha256"] = sha256_file(Path(manifest_path))
    payload.setdefault("created_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return schema.write_json(Path(meta_path), payload, "split_meta")


def write_norm_stats(
    stats_path: Path,
    mean: Sequence[float],
    std: Sequence[float],
    num_images: int,
) -> Path:
    """Write ``norm_stats.json`` - per-channel statistics of the training images.

    ``fitted_on`` is always ``"train"``: statistics computed on validation or
    test images would leak held-out information into training.
    """
    payload = _norm_stats_payload(mean, std, num_images)
    return schema.write_json(Path(stats_path), payload, "norm_stats")


def _norm_stats_payload(
    mean: Sequence[float], std: Sequence[float], num_images: int
) -> dict[str, Any]:
    return {
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "fitted_on": "train",
        "num_images": int(num_images),
    }
