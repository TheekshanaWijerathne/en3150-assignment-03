"""Resource profiling - what each model costs to store and to run.

The cost side of the accuracy / memory / computation trade-off in Sections 4
and 6. Every model is measured by the same code, in one session, on one
computer: a comparison table built from numbers measured in different ways
cannot be defended.

For each architecture:

* **parameters** - trainable and total;
* **size on disk** - the saved weights (``state_dict``), in KB
  (1 KB = 1,024 bytes);
* **MACs** - multiply-accumulates for one image (:func:`count_macs`);
* **latency** - milliseconds per image on one CPU thread, at batch size 1, as
  an edge device would run it;
* **peak memory** - the weights plus the largest set of intermediate results
  alive at the same time during one forward pass, in MB (1 MB = 1,048,576 bytes).

None of these depend on trained weights - an architecture costs the same with
random weights as with trained ones - so models are built fresh, and no
checkpoints are needed.

Team notes:
Owner: Member 1. profile_all runs in notebooks/06_resource_benchmark.ipynb;
count_macs is also used in notebook 02.

    IN   run_ids + their experiment configs (architecture only)
         history.json  (for mean_epoch_time_s)
    OUT  ResourceProfile per run - always returned
         -> results/metrics/<run_id>/resources.json   (official only)
"""

from __future__ import annotations

import itertools
import statistics
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from edgecnn.utils.io import repo_relative
from edgecnn.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch
    from torch import nn

    from edgecnn.contracts.types import ResolvedConfig, ResourceProfile

log = get_logger(__name__)

#: CPU threads used to measure latency. One: a single-core edge device has no
#: more, and one thread gives far steadier timings than a busy multi-core laptop.
LATENCY_THREADS = 1

#: How peak memory was found, recorded in resources.json -> peak_mem_method.
PEAK_FROM_GRAPH = "graph"
PEAK_PER_LAYER = "per-layer estimate"


def profile_model(
    model: nn.Module,
    input_shape: tuple[int, int, int],
    run_id: str,
    model_name: str,
    mean_epoch_time_s: float = 0.0,
    latency_device: str = "cpu",
    latency_repeats: int = 100,
    latency_warmup: int = 10,
) -> ResourceProfile:
    """Measure one model: parameters, size, MACs, latency and peak memory.

    The model itself is never changed: every measurement works on a copy.

    Args:
        model: The architecture to measure; its weights do not matter.
        input_shape: ``(channels, height, width)`` of one image.
        run_id: The run the profile is for.
        model_name: Its model key, e.g. ``"model_b"``.
        mean_epoch_time_s: Training time per epoch, copied from the run's
            ``history.json`` - it is never re-measured here.
        latency_device: Where latency is measured - the CPU for every model,
            since edge devices have no GPU.
        latency_repeats: Timed forward passes; their mean is the latency.
        latency_warmup: Untimed passes first.

    Returns:
        The :class:`~edgecnn.contracts.types.ResourceProfile`.
    """
    profile, _ = _measure(
        model, input_shape, run_id=run_id, model_name=model_name,
        mean_epoch_time_s=mean_epoch_time_s, device=latency_device,
        repeats=latency_repeats, warmup=latency_warmup,
    )
    return profile


def count_macs(model: nn.Module, input_shape: tuple[int, int, int]) -> int:
    """Multiply-accumulate operations (MACs) for one image's forward pass.

    Counts the convolutions and fully-connected layers - the convention of the
    MobileNet and SqueezeNet papers and of torchvision's published model
    figures, which this reproduces exactly (MobileNetV2 0.301 G, SqueezeNet 1.1
    0.349 G, at 224x224). Batch-norm, activations, pooling and bias additions
    are left out: they are cheap next to the convolutions, and counting them
    would make the numbers incomparable with published ones.

    * a convolution costs output values x (input channels / groups) x kernel
      area - so a depthwise convolution, with one group per channel, costs a
      fraction of a standard one;
    * a fully-connected layer costs output values x input features.

    One MAC is one multiply and one add; papers that count FLOPs report
    roughly twice as many. The model is measured on a copy in evaluation mode,
    so the caller's model - its weights, batch-norm statistics and train /
    eval mode - is untouched.

    Args:
        model: Any model that takes a ``(1, *input_shape)`` batch.
        input_shape: ``(channels, height, width)`` of one image.

    Returns:
        The number of MACs at batch size 1.
    """
    import copy
    import math

    import torch
    from torch import nn

    probe = copy.deepcopy(model).cpu().eval()
    total = 0

    def conv(module: Any, inputs: Any, output: torch.Tensor) -> None:
        nonlocal total
        per_value = (module.in_channels // module.groups) * math.prod(module.kernel_size)
        total += output[0].numel() * per_value

    def linear(module: Any, inputs: Any, output: torch.Tensor) -> None:
        nonlocal total
        total += output[0].numel() * module.in_features

    hooks = []
    for module in probe.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            hooks.append(module.register_forward_hook(conv))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear))
    try:
        with torch.no_grad():
            probe(torch.zeros(1, *input_shape))
    finally:
        for hook in hooks:
            hook.remove()
    return int(total)


def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, int, int],
    device: str = "cpu",
    batch_size: int = 1,
    warmup: int = 10,
    repeats: int = 100,
) -> float:
    """Mean milliseconds per forward pass, measured on a copy of the model.

    On a CPU the passes run on ``LATENCY_THREADS`` (one) thread, and the
    previous thread setting is restored afterwards. The first ``warmup`` passes
    are not timed: they pay one-off costs, such as allocating memory and
    choosing algorithms, that would inflate the result. On a GPU every pass is
    synchronised before the clock stops, because GPU work runs asynchronously.

    Raises:
        ValueError: if ``repeats`` is less than 1.
    """
    return statistics.fmean(
        _latency_samples_ms(model, input_shape, device, batch_size, warmup, repeats)
    )


def measure_model_size_kb(model: nn.Module) -> float:
    """Size of the saved weights (``state_dict``) in KB, where 1 KB = 1,024 bytes.

    Measured by saving them with ``torch.save``, exactly as a deployment would
    store them, rather than computed as parameters x 4 bytes: the saved weights
    also include buffers such as batch-norm statistics, which take real bytes
    on the device. This reproduces torchvision's published file sizes to within
    0.3%. It is never measured from a training checkpoint, which also holds the
    optimizer's state - several times the size of the weights.
    """
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1024


def measure_peak_memory_mb(model: nn.Module, input_shape: tuple[int, int, int]) -> float:
    """Peak memory of one forward pass at batch size 1, in MB (1 MB = 1,048,576 bytes).

    The weights (parameters and buffers) plus the largest set of intermediate
    results alive at the same time. It is worked out from the model's graph
    (``torch.fx``): each intermediate result is freed after the last operation
    that uses it, and results that share memory - views and in-place
    operations - are counted once. The result is the minimum a perfect memory
    manager would need, independent of the machine and the runtime.

    A model whose graph cannot be traced gets a per-layer estimate instead: the
    largest inputs-plus-outputs of any single layer, plus the weights. That
    misses results kept alive across layers, such as skip connections, so
    ``profile_all`` records which method was used.
    """
    return _peak_memory(model, input_shape)[0]


def profile_all(
    run_ids: list[str],
    *,
    mode: str = "official",
    latency_device: str = "cpu",
) -> dict[str, ResourceProfile]:
    """Profile every run's architecture in one session - notebook 06.

    For each run, loads its experiment config in ``mode``, builds the model with
    ``build_model_from_config`` - the same builder and settings its training
    used, with fresh weights - and measures it, taking the mean epoch time from
    the run's ``history.json``. Runs that share an architecture (the three
    ``model_b`` optimizer runs) share one measurement; only their epoch times
    differ. Only if the config is official, each run's profile is written to
    ``results/metrics/<run_id>/resources.json``.

    Outside official mode, a run whose model is not implemented yet is skipped
    with a warning, so the notebook can be developed before every model exists.

    Returns:
        ``{run_id: ResourceProfile}`` in every mode, in the order given.

    Raises:
        ValueError: if a run id does not match its experiment config.
        FileNotFoundError: in official mode, if a run has no ``history.json``
            yet - its training notebook must run officially first.
        NotImplementedError: in official mode, if a model is still a stub.
    """
    import dataclasses
    import json

    from edgecnn.config.loader import load_config
    from edgecnn.contracts import paths, schema
    from edgecnn.models.registry import (
        build_model_from_config,
        model_input_shape,
        model_settings,
    )

    profiles: dict[str, ResourceProfile] = {}
    measured: dict[tuple[Any, ...], tuple[ResourceProfile, dict[str, Any]]] = {}
    for run_id in run_ids:
        cfg = load_config(paths.experiment_config(run_id), mode=mode)
        if cfg.run_id != run_id:
            raise ValueError(f"{run_id!r} does not match its experiment config ({cfg.run_id!r})")
        input_shape = model_input_shape(cfg)
        settings = json.dumps(model_settings(cfg), sort_keys=True, default=str)
        architecture = (cfg.model_name, input_shape, settings)
        if architecture not in measured:
            try:
                model = build_model_from_config(cfg, _num_classes(cfg))
            except NotImplementedError as exc:
                if cfg.is_official:
                    raise
                log.warning("Skipped %s: its model is not implemented yet (%s)", run_id, exc)
                continue
            benchmark = cfg.section("benchmark")
            measured[architecture] = _measure(
                model, input_shape, run_id=run_id, model_name=cfg.model_name,
                mean_epoch_time_s=0.0, device=latency_device,
                repeats=int(benchmark.get("latency_repeats", 100)),
                warmup=int(benchmark.get("latency_warmup", 10)),
            )
        shared, extras = measured[architecture]
        profile = dataclasses.replace(
            shared, run_id=run_id, model_name=cfg.model_name,
            mean_epoch_time_s=_mean_epoch_time_s(cfg),
        )
        profiles[run_id] = profile
        if cfg.is_official:
            schema.write_json(
                paths.resources_json(run_id), _resources_payload(profile, extras), "resources"
            )
    return profiles


# --- internals --------------------------------------------------------------------


def _measure(
    model: nn.Module,
    input_shape: tuple[int, int, int],
    *,
    run_id: str,
    model_name: str,
    mean_epoch_time_s: float,
    device: str,
    repeats: int,
    warmup: int,
) -> tuple[ResourceProfile, dict[str, Any]]:
    """A model's profile, plus the details resources.json records beyond it."""
    from edgecnn.contracts.types import ResourceProfile
    from edgecnn.utils.device import describe_device

    samples = _latency_samples_ms(model, input_shape, device, 1, warmup, repeats)
    peak_mb, method = _peak_memory(model, input_shape)
    profile = ResourceProfile(
        run_id=run_id,
        model_name=model_name,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_params=sum(p.numel() for p in model.parameters()),
        model_size_kb=measure_model_size_kb(model),
        macs=count_macs(model, input_shape),
        mean_epoch_time_s=float(mean_epoch_time_s),
        inference_latency_ms=statistics.fmean(samples),
        peak_mem_mb=peak_mb,
        device=describe_device(device),
    )
    extras = {
        "input_shape": list(input_shape),
        "latency_median_ms": statistics.median(samples),
        "latency_batch_size": 1,
        "latency_repeats": repeats,
        "latency_threads": LATENCY_THREADS if str(device).startswith("cpu") else None,
        "peak_mem_method": method,
    }
    return profile, extras


def _latency_samples_ms(
    model: nn.Module,
    input_shape: tuple[int, int, int],
    device: str,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> list[float]:
    """Milliseconds of each timed forward pass; see :func:`measure_latency`."""
    import copy
    import time

    import torch

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    target = torch.device(device)
    probe = copy.deepcopy(model).to(target).eval()
    generator = torch.Generator().manual_seed(0)
    batch = torch.randn(batch_size, *input_shape, generator=generator).to(target)

    def synchronize() -> None:
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    threads = torch.get_num_threads()
    if target.type == "cpu":
        torch.set_num_threads(LATENCY_THREADS)
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                probe(batch)
            synchronize()
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                probe(batch)
                synchronize()
                samples.append((time.perf_counter() - started) * 1000)
    finally:
        torch.set_num_threads(threads)
    return samples


def _peak_memory(model: nn.Module, input_shape: tuple[int, int, int]) -> tuple[float, str]:
    """``(peak MB, method)``; see :func:`measure_peak_memory_mb`."""
    import copy

    probe = copy.deepcopy(model).cpu().eval()
    try:
        activations, method = max(_activation_timeline(probe, input_shape)), PEAK_FROM_GRAPH
    except Exception as exc:  # torch.fx cannot trace every model, e.g. data-dependent branches
        log.warning(
            "Peak memory of %s is a per-layer estimate: its graph could not be traced (%s)",
            type(model).__name__, exc,
        )
        activations, method = _per_layer_peak_bytes(probe, input_shape), PEAK_PER_LAYER
    return (_weight_bytes(probe) + activations) / 2**20, method


def _activation_timeline(model: nn.Module, input_shape: tuple[int, int, int]) -> list[int]:
    """Bytes of intermediate results alive while each node of the model's graph runs.

    ``model`` should be a CPU copy in evaluation mode. Every value is kept until
    the run ends, so no memory address is reused while it runs: that is what
    lets results that share memory - views, in-place operations - be recognised
    by their address. A result counts from the step that creates it to the last
    step that uses it; the model's output, to the end.
    """
    import torch
    from torch import fx

    graph_module = fx.symbolic_trace(model)
    nodes = list(graph_module.graph.nodes)
    position = {node: index for index, node in enumerate(nodes)}
    values: dict[Any, Any] = {}

    class Recorder(fx.Interpreter):
        def run_node(self, node: Any) -> Any:
            value = super().run_node(node)
            values[node] = value
            return value

    with torch.no_grad():
        Recorder(graph_module, garbage_collect_values=False).run(torch.zeros(1, *input_shape))

    weights = {
        tensor.untyped_storage().data_ptr()
        for tensor in itertools.chain(graph_module.parameters(), graph_module.buffers())
    }
    storages: dict[int, list[int]] = {}  # address -> [bytes, first step, last step]
    end = len(nodes) - 1
    for node in nodes:
        if node.op == "get_attr":  # weights and constants, counted with the weights
            continue
        last = end if node.op == "output" else max(
            (position[user] for user in node.users), default=position[node]
        )
        for tensor in _tensors(values.get(node)):
            storage = tensor.untyped_storage()
            address = storage.data_ptr()
            if address in weights:
                continue
            if address in storages:  # a view or in-place result: the same memory
                storages[address][2] = max(storages[address][2], last)
            else:
                storages[address] = [storage.nbytes(), position[node], last]
    return [
        sum(size for size, first, final in storages.values() if first <= step <= final)
        for step in range(len(nodes))
    ]


def _per_layer_peak_bytes(model: nn.Module, input_shape: tuple[int, int, int]) -> int:
    """The largest inputs-plus-outputs of any single layer - the fallback estimate."""
    import torch

    peak = 0

    def record(module: Any, inputs: Any, output: Any) -> None:
        nonlocal peak
        storages = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for tensor in itertools.chain(_tensors(inputs), _tensors(output))
        }
        peak = max(peak, sum(storages.values()))

    layers = [module for module in model.modules() if next(module.children(), None) is None]
    hooks = [layer.register_forward_hook(record) for layer in layers]
    try:
        with torch.no_grad():
            model(torch.zeros(1, *input_shape))
    finally:
        for hook in hooks:
            hook.remove()
    return peak


def _weight_bytes(model: nn.Module) -> int:
    """Bytes of the parameters and buffers, each memory block counted once."""
    storages = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for tensor in itertools.chain(model.parameters(), model.buffers())
    }
    return sum(storages.values())


def _tensors(value: Any) -> Iterator[torch.Tensor]:
    """Every tensor inside a value, however nested in tuples, lists or dicts."""
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def _num_classes(cfg: ResolvedConfig) -> int:
    """Classes in the run's dataset, from its split_meta.json."""
    from edgecnn.contracts import paths, schema

    if cfg.section("dataset").get("name") == "synthetic":
        from edgecnn.data.synthetic import fixture_is_current, make_synthetic_fixture

        root = paths.SYNTHETIC_FIXTURE_DIR
        if not fixture_is_current(root):
            make_synthetic_fixture(root)
        return int(schema.read_json(root / "split_meta.json", "split_meta")["num_classes"])
    return int(schema.read_json(paths.SPLIT_META, "split_meta")["num_classes"])


def _mean_epoch_time_s(cfg: ResolvedConfig) -> float:
    """Mean seconds per training epoch, from the run's history.json; 0 if it has none."""
    from edgecnn.contracts import paths, schema

    path = paths.history_json(cfg.run_id)
    if not path.exists():
        if cfg.is_official:
            raise FileNotFoundError(
                f"{repo_relative(path)} is missing: run this run's training notebook (03, 04 "
                "or 05) in official mode and commit its results first"
            )
        return 0.0
    epochs = schema.read_json(path, "history")["epochs"]
    return statistics.fmean(float(epoch["epoch_time_s"]) for epoch in epochs)


def _resources_payload(profile: ResourceProfile, extras: dict[str, Any]) -> dict[str, Any]:
    """The contents of ``resources.json`` for one run."""
    return {
        "run_id": profile.run_id,
        "model": profile.model_name,
        "trainable_params": profile.trainable_params,
        "total_params": profile.total_params,
        "model_size_kb": profile.model_size_kb,
        "macs": profile.macs,
        "input_shape": extras["input_shape"],
        "mean_epoch_time_s": profile.mean_epoch_time_s,
        "inference_latency_ms": profile.inference_latency_ms,
        "latency_median_ms": extras["latency_median_ms"],
        "latency_batch_size": extras["latency_batch_size"],
        "latency_repeats": extras["latency_repeats"],
        "latency_threads": extras["latency_threads"],
        "peak_mem_mb": profile.peak_mem_mb,
        "peak_mem_method": extras["peak_mem_method"],
        "device": profile.device,
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
