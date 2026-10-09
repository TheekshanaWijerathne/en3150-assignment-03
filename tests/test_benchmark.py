"""Unit tests for resource profiling: size on disk, latency, peak memory and profile_all."""

from __future__ import annotations

import pytest

from edgecnn.config.loader import load_config
from edgecnn.contracts import paths, schema

torch = pytest.importorskip("torch", reason="torch not installed")

from torch import nn  # noqa: E402

from edgecnn.evaluation.benchmark import (  # noqa: E402
    PEAK_FROM_GRAPH,
    PEAK_PER_LAYER,
    _activation_timeline,
    _peak_memory,
    _weight_bytes,
    count_macs,
    measure_latency,
    measure_model_size_kb,
    measure_peak_memory_mb,
    profile_all,
)

MIB = 2**20


def _experiment(name: str) -> str:
    return str(paths.CONFIGS_DIR / "experiments" / f"{name}.yaml")


# --- size on disk ----------------------------------------------------------------


def test_size_is_the_saved_weights(tmp_path) -> None:
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8))
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    assert measure_model_size_kb(model) * 1024 == (tmp_path / "weights.pt").stat().st_size


def test_sizes_match_torchvisions_published_file_sizes() -> None:
    from torchvision import models

    for name, weights in (
        ("mobilenet_v2", models.MobileNet_V2_Weights.IMAGENET1K_V1),
        ("squeezenet1_1", models.SqueezeNet1_1_Weights.IMAGENET1K_V1),
    ):
        size_mb = measure_model_size_kb(getattr(models, name)(weights=None)) / 1024
        assert size_mb == pytest.approx(weights.meta["_file_size"], rel=0.01)


# --- latency ---------------------------------------------------------------------


class _ThreadSpy(nn.Module):
    """Records how many CPU threads PyTorch may use while it runs."""

    seen: list[int] = []  # on the class, so the copy that gets timed shares it

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ThreadSpy.seen.append(torch.get_num_threads())
        return self.linear(x)


def test_latency_runs_on_one_thread_and_restores_the_setting() -> None:
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        _ThreadSpy.seen.clear()
        assert measure_latency(_ThreadSpy(), (4,), warmup=1, repeats=3) > 0
        assert set(_ThreadSpy.seen) == {1}
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(original)


def test_latency_leaves_the_model_untouched_and_needs_a_repeat() -> None:
    model = nn.Linear(4, 2).train()
    measure_latency(model, (4,), warmup=0, repeats=1)
    assert model.training
    with pytest.raises(ValueError):
        measure_latency(model, (4,), repeats=0)


# --- peak memory -----------------------------------------------------------------


class _Residual(nn.Module):
    """x + conv2(relu(conv1(x))): the input stays alive across the whole block."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


def test_peak_memory_follows_each_result_until_its_last_use() -> None:
    # every result is 4 x 8 x 8 float32 values = 1,024 bytes
    # steps: input, conv1, relu, conv2, add, output
    model = _Residual().eval()
    assert _activation_timeline(model, (4, 8, 8)) == [1024, 2048, 3072, 3072, 3072, 1024]
    weights = 2 * 4 * 4 * 3 * 3 * 4  # two 3x3 convolutions, 4 -> 4 channels
    assert measure_peak_memory_mb(model, (4, 8, 8)) == (3072 + weights) / MIB


def test_in_place_results_and_views_share_memory() -> None:
    model = nn.Sequential(
        nn.Conv2d(4, 4, 3, padding=1, bias=False), nn.ReLU(inplace=True), nn.Flatten()
    ).eval()
    # steps: input, conv, in-place relu, flatten (a view), output
    assert _activation_timeline(model, (4, 8, 8)) == [1024, 2048, 1024, 1024, 1024]


class _Branchy(nn.Module):
    """Branches on its data, which a graph tracer cannot follow."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y if y.sum() > 0 else -y


def test_an_untraceable_model_gets_a_per_layer_estimate() -> None:
    peak_mb, method = _peak_memory(_Branchy(), (3, 8, 8))
    assert method == PEAK_PER_LAYER
    layer = 3 * 8 * 8 * 4 + 4 * 8 * 8 * 4  # the convolution's input plus its output
    weights = (4 * 3 * 3 * 3 + 4) * 4
    assert peak_mb == (layer + weights) / MIB


def test_every_architecture_is_measured_from_its_graph() -> None:
    from torchvision import models

    from edgecnn.models.registry import build_model_from_config

    candidates = [
        build_model_from_config(load_config(_experiment(name), mode="synthetic"), 10)
        for name in ("model_a__adam", "model_b__adam")
    ]
    candidates += [  # torchvision's own copies, standing in until the pretrained builders exist
        models.mobilenet_v2(weights=None, num_classes=10),
        models.squeezenet1_1(weights=None, num_classes=10),
    ]
    for model in candidates:
        peak_mb, method = _peak_memory(model, (3, 64, 64))
        assert method == PEAK_FROM_GRAPH
        assert peak_mb > _weight_bytes(model) / MIB


# --- profile_all -----------------------------------------------------------------


def test_runs_sharing_an_architecture_share_one_measurement(monkeypatch, tmp_path) -> None:
    from edgecnn.models.registry import build_model_from_config

    monkeypatch.setattr(paths, "METRICS_DIR", tmp_path / "metrics")
    runs = ["model_b__adam__seed42", "model_b__sgd__seed42"]
    adam, sgd = profile_all(runs, mode="synthetic").values()
    assert (adam.run_id, sgd.run_id) == tuple(runs)
    assert adam.inference_latency_ms == sgd.inference_latency_ms

    model = build_model_from_config(load_config(_experiment("model_b__adam"), mode="synthetic"), 10)
    assert adam.total_params == sum(p.numel() for p in model.parameters())
    assert adam.macs == count_macs(model, (3, 64, 64))
    assert adam.mean_epoch_time_s == 0.0 and adam.device.startswith("cpu")
    assert not (tmp_path / "metrics").exists()  # synthetic runs never write


def test_unfinished_models_are_skipped_outside_official_mode(monkeypatch) -> None:
    from edgecnn.models import registry

    real = registry.build_model_from_config

    def build(cfg, num_classes):
        if cfg.model_name == "model_a":
            raise NotImplementedError("Model A is still a stub")
        return real(cfg, num_classes)

    monkeypatch.setattr(registry, "build_model_from_config", build)
    profiles = profile_all(["model_a__adam__seed42", "model_b__adam__seed42"], mode="synthetic")
    assert list(profiles) == ["model_b__adam__seed42"]


def test_official_profiles_record_the_training_time(monkeypatch, tmp_path, valid_history) -> None:
    run = "model_b__adam__seed42"
    monkeypatch.setattr(paths, "METRICS_DIR", tmp_path / "metrics")
    schema.write_json(paths.history_json(run), valid_history, "history")

    profile = profile_all([run], mode="official")[run]
    written = schema.read_json(paths.resources_json(run), "resources")
    assert written["mean_epoch_time_s"] == profile.mean_epoch_time_s == pytest.approx(12.5)
    assert written["input_shape"] == [3, 64, 64] and written["latency_threads"] == 1
    assert written["peak_mem_method"] == PEAK_FROM_GRAPH
    assert written["latency_median_ms"] > 0 and written["device"].startswith("cpu")


def test_official_profiling_needs_the_training_history(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(paths, "METRICS_DIR", tmp_path / "metrics")
    with pytest.raises(FileNotFoundError, match="history.json"):
        profile_all(["model_b__adam__seed42"], mode="official")


def test_a_run_id_must_match_its_config() -> None:
    with pytest.raises(ValueError, match="seed7"):
        profile_all(["model_b__adam__seed7"], mode="synthetic")
