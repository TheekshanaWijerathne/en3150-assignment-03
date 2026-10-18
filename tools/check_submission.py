"""Check that the submission's code computes exactly what the development code computed.

    python tools/check_submission.py

Runs the code cells of the submission's shared parts, as the notebooks will, next to the edgecnn
library that produced the reported results, and compares them on the real dataset (data/raw must
hold EuroSAT): the split and its fingerprint, the decoded pixels and normalisation statistics, the
first training batches, every model's initial weights, a short training run weight for weight, the
metrics, every cost measurement, and the architecture and comparison tables against the committed
ones. Training is compared on the CPU, on a small subset, so the check takes a few minutes.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_submission  # noqa: E402

from edgecnn.config import load_config  # noqa: E402
from edgecnn.contracts import paths  # noqa: E402

ROOT = build_submission.ROOT
PARTS = [
    "common",
    "data",
    "dataset_figures",
    "custom_models",
    "pretrained",
    "macs",
    "architecture_tables",
    "training",
    "evaluation",
    "optimizer_study",
    "benchmark",
    "reporting",
]
RUN_NAMES = {  # development run id -> submission run name
    "model_b__sgd__seed42": "model_b_sgd",
    "model_b__sgd_momentum__seed42": "model_b_sgd_momentum",
    "model_b__adam__seed42": "model_b_adam",
    "model_a__adam__seed42": "model_a_adam",
    "mobilenet_v2__adam__seed42": "mobilenet_v2_adam",
    "squeezenet1_1__adam__seed42": "squeezenet1_1_adam",
}
failures: list[str] = []


def check(condition: bool, what: str) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {what}", flush=True)
    if not condition:
        failures.append(what)


def load_parts(workdir: Path) -> dict:
    """Run every part's code cells, in order, in one namespace, from ``workdir``."""
    module = types.ModuleType("submission_parts")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    previous = Path.cwd()
    os.chdir(workdir)
    try:
        for name in PARTS:
            for kind, source in build_submission.parse(build_submission.PARTS / f"{name}.md"):
                if kind == "code":
                    code = "\n".join(
                        line for line in source.splitlines() if not line.lstrip().startswith("%")
                    )
                    exec(compile(code, f"<part {name}>", "exec"), namespace)
    finally:
        os.chdir(previous)
    return namespace


def config(name: str, mode: str = "official"):
    return load_config(paths.CONFIGS_DIR / "experiments" / f"{name}.yaml", mode=mode)


def same_state(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    first, second = a.state_dict(), b.state_dict()
    return list(first) == list(second) and all(
        torch.equal(first[k].cpu(), second[k].cpu()) for k in first
    )


def main() -> int:
    from edgecnn.data import build_dataloaders
    from edgecnn.evaluation.benchmark import (
        count_macs,
        measure_model_size_kb,
        measure_peak_memory_mb,
    )
    from edgecnn.evaluation.metrics import _test_metrics_payload, compute_metrics
    from edgecnn.models.registry import build_model_from_config
    from edgecnn.training import Trainer

    workdir = Path(tempfile.mkdtemp(prefix="submission-check-"))
    print("Running the submission's parts in a temporary folder\n")
    sub = load_parts(workdir)
    sub["RESULTS_DIR"] = workdir / "results"
    sub["CHECKPOINT_DIR"] = workdir / "checkpoints"
    torch.hub.set_dir(str(workdir / "torch_hub"))  # ImageNet weights: into the temporary folder

    # --- data -----------------------------------------------------------------------------
    print("Data")
    sub["DATA_DIR"] = paths.RAW_DIR  # the development download, so nothing is fetched again
    classes, rows = sub["download_eurosat"]()
    labels = [row["label_index"] for row in rows]
    for row, split in zip(
        rows, sub["stratified_split"](labels, sub["SPLIT_FRACTIONS"], sub["SEED"]), strict=True
    ):
        row["split"] = split
    fingerprint = sub["split_fingerprint"](rows)
    meta = json.loads(paths.SPLIT_META.read_text(encoding="utf-8"))
    check(
        fingerprint == meta["manifest_sha256"] == sub["EXPECTED_SPLIT_SHA256"],
        "the split's fingerprint equals the committed split's",
    )
    pixels = sub["load_pixels"](rows, workdir / "pixels.npy")
    (dev_cache,) = paths.PROCESSED_DIR.glob("eurosat_64.pixels-*.npy")
    check(
        np.array_equal(pixels, np.load(dev_cache)),
        "every decoded pixel equals the development cache",
    )
    split_of = np.array([row["split"] for row in rows])
    splits = {name: np.flatnonzero(split_of == name) for name in sub["SPLIT_NAMES"]}
    mean, std = sub["channel_stats"](pixels, splits["train"])
    committed = json.loads(paths.NORM_STATS.read_text(encoding="utf-8"))
    check(
        mean == committed["mean"] and std == committed["std"],
        "the normalisation statistics equal the committed ones, bit for bit",
    )
    data = sub["EuroSATData"](
        classes, rows, np.array(labels, dtype=np.int64), splits, pixels, mean, std, fingerprint
    )

    print("Batches")
    dev = build_dataloaders(config("model_b__adam"))
    loaders = sub["make_loaders"](data, mean, std)
    for split in ("train", "val"):
        torch.manual_seed(42)
        dev_batches = [batch for _, batch in zip(range(3), getattr(dev, split), strict=False)]
        torch.manual_seed(42)
        sub_batches = [batch for _, batch in zip(range(3), loaders[split], strict=False)]
        check(
            all(
                torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                for a, b in zip(dev_batches, sub_batches, strict=True)
            ),
            f"the first three {split} batches are identical, augmentation included",
        )
    pretrained_cfg = config("mobilenet_v2__adam")
    dev_imagenet = build_dataloaders(pretrained_cfg)
    imagenet = sub["make_loaders"](data, sub["IMAGENET_MEAN"], sub["IMAGENET_STD"])
    torch.manual_seed(42)
    first_dev = next(iter(dev_imagenet.train))
    torch.manual_seed(42)
    first_sub = next(iter(imagenet["train"]))
    check(torch.equal(first_dev[0], first_sub[0]), "the ImageNet-normalised batches are identical")

    # --- models ---------------------------------------------------------------------------
    print("Models")
    for name in ("model_a", "model_b", "mobilenet_v2", "squeezenet1_1"):
        dev_model = build_model_from_config(config(f"{name}__adam"), 10)
        sub_model = sub["build_model"](name)
        flags = [p.requires_grad for p in dev_model.parameters()] == [
            p.requires_grad for p in sub_model.parameters()
        ]
        check(
            same_state(dev_model, sub_model) and flags,
            f"{name}: identical initial weights and frozen layers",
        )

    # --- training -------------------------------------------------------------------------
    print("Training (CPU, 320 training and 128 validation images, 2 epochs)")
    small = dataclasses.replace(
        data,
        splits={
            "train": splits["train"][:320],
            "val": splits["val"][:128],
            "test": splits["test"][:128],
        },
    )
    sub["DEVICE"] = torch.device("cpu")
    sub["EPOCHS"] = 2
    for experiment, name, optimizer in (
        ("model_b__sgd_momentum", "model_b", "sgd_momentum"),
        ("model_b__adam", "model_b", "adam"),
        ("mobilenet_v2__adam", "mobilenet_v2", "adam"),
    ):
        run = f"{name}_{optimizer}"
        imagenet_stats = name == "mobilenet_v2"
        stats = (sub["IMAGENET_MEAN"], sub["IMAGENET_STD"]) if imagenet_stats else (mean, std)

        cfg = dataclasses.replace(config(experiment, mode="debug"), device="cpu")
        dev_loaders = sub["make_loaders"](small, *stats)
        bundle = SimpleNamespace(
            train=dev_loaders["train"],
            val=dev_loaders["val"],
            class_names=classes,
            input_shape=(3, 64, 64),
        )
        dev_model = build_model_from_config(cfg, 10)
        dev_result = Trainer(cfg).fit(dev_model, bundle, cfg)

        sub_model = sub["build_model"](name)
        scale = (
            sub["PRETRAINED_SETTINGS"]["mobilenet_v2"]["backbone_lr_scale"]
            if imagenet_stats
            else None
        )
        history = sub["train"](
            sub_model, name, optimizer, run, sub["make_loaders"](small, *stats), scale
        )
        dev_epochs = [
            (e.train_loss, e.val_loss, e.train_acc, e.val_acc, e.lr) for e in dev_result.epochs
        ]
        sub_epochs = [
            (e["train_loss"], e["val_loss"], e["train_acc"], e["val_acc"], e["lr"])
            for e in history["epochs"]
        ]
        check(
            dev_epochs == sub_epochs and same_state(dev_model, sub_model),
            f"{run}: identical losses, accuracies, learning rates and final weights",
        )

    # --- metrics --------------------------------------------------------------------------
    print("Metrics")
    rng = np.random.default_rng(0)
    y_true, y_pred = rng.integers(0, 10, 500), rng.integers(0, 9, 500)  # class 9 never predicted
    dev_payload = _test_metrics_payload(
        compute_metrics(y_true, y_pred, classes, "x", "model_b"), None
    )
    sub_metrics = sub["compute_metrics"](y_true, y_pred, classes)
    keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "per_class",
        "confusion_matrix",
        "num_samples",
    ]
    check(all(dev_payload[k] == sub_metrics[k] for k in keys), "every metric is identical")

    # --- costs ----------------------------------------------------------------------------
    print("Costs")
    for run_id, run in RUN_NAMES.items():
        if not run.endswith("adam"):
            continue
        name = run[: -len("_adam")]
        committed = json.loads(paths.resources_json(run_id).read_text(encoding="utf-8"))
        model = sub["build_model"](name)
        values = {
            "trainable_params": sub["count_parameters"](model),
            "total_params": sub["count_parameters"](model, trainable_only=False),
            "model_size_kb": sub["model_size_kb"](model),
            "macs": sub["count_macs"](model),
            "peak_mem_mb": sub["peak_memory_mb"](model),
        }
        dev_values = {
            "model_size_kb": measure_model_size_kb(model),
            "macs": count_macs(model, (3, 64, 64)),
            "peak_mem_mb": measure_peak_memory_mb(model, (3, 64, 64)),
        }
        check(
            all(values[k] == committed[k] for k in values)
            and all(values[k] == dev_values[k] for k in dev_values),
            f"{name}: parameters, size, MACs and peak memory equal the committed measurements",
        )

    # --- tables ---------------------------------------------------------------------------
    print("Tables")
    custom, breakdown = sub["architecture_tables"](
        sub["build_model"]("model_a"), sub["build_model"]("model_b")
    )
    check(
        custom == paths.CUSTOM_ARCHITECTURES_TABLE.read_text(encoding="utf-8"),
        "the architecture table equals the committed one",
    )
    check(
        breakdown == paths.PARAM_BREAKDOWN_TABLE.read_text(encoding="utf-8"),
        "the parameter breakdown equals the committed one",
    )
    benchmark = {}
    for run_id, run in RUN_NAMES.items():
        folder = sub["RESULTS_DIR"] / run
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(paths.history_json(run_id), folder / "history.json")
        shutil.copy(paths.test_metrics_json(run_id), folder / "test_metrics.json")
        resources = json.loads(paths.resources_json(run_id).read_text(encoding="utf-8"))
        benchmark[resources["model"]] = {
            "trainable_params": resources["trainable_params"],
            "total_params": resources["total_params"],
            "model_size_kb": resources["model_size_kb"],
            "macs": resources["macs"],
            "latency_ms": resources["inference_latency_ms"],
            "peak_memory_mb": resources["peak_mem_mb"],
        }
    for function, table in (
        ("optimizer_table", paths.OPTIMIZER_COMPARISON_TABLE),
        ("custom_model_table", paths.CUSTOM_COMPARISON_TABLE),
        ("final_table", paths.FINAL_COMPARISON_TABLE),
    ):
        made = sub[function]() if function == "optimizer_table" else sub[function](benchmark)
        check(made == table.read_text(encoding="utf-8"), f"{table.name} equals the committed table")

    shutil.rmtree(workdir, ignore_errors=True)
    print(f"\n{'All checks passed.' if not failures else f'{len(failures)} check(s) FAILED.'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
