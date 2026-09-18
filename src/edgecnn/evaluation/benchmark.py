"""SEAM 4 - resource profiling.          Assignment Sections 4 [25] + 6 [20]

Owner: Member 1.  Used for EVERY model.

    +---------------------------------------------------------------------+
    |  IN   model : nn.Module                                              |
    |       input_shape : (3, 64, 64)                                      |
    |       history.json  (for mean_epoch_time_s)                          |
    |  OUT  ResourceProfile -> results/metrics/<run_id>/resources.json     |
    +---------------------------------------------------------------------+

This module produces the cost side of the accuracy/memory/compute trade-off
that Section 6 is entirely about. Every number here must come from one code
path, run on one device, or the comparison is not a comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

    from edgecnn.contracts.types import ResourceProfile


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
    """Measure parameters, size, MACs, latency and peak memory.

    What each number means, and the trap in each:

    * **trainable_params** - ``requires_grad=True`` only. Equals total for the
      custom models; smaller for a partially frozen backbone.
    * **total_params** - everything. *This*, not the trainable count,
      determines the memory an edge device needs at inference. Section 5 asks
      for both, and conflating them flatters the fine-tuned models.
    * **model_size_kb** - serialise the ``state_dict`` to a temp file and
      measure it, rather than computing ``numel * 4``. The computed figure
      misses buffers such as BatchNorm running statistics, which are real
      bytes on the device.
    * **macs** - via ``thop.profile``. Note that thop reports MACs while some
      papers report FLOPs at roughly ``2 x MACs``; state which is used in the
      report or the comparison against published MobileNet figures will look
      wrong by a factor of two.
    * **inference_latency_ms** - batch size 1, ``model.eval()``,
      ``torch.no_grad()``, warm up first. Measure on **CPU for every model**
      even on a CUDA machine: the Section 6 argument is about edge deployment,
      and edge devices have no GPU. On GPU you would also need
      ``torch.cuda.synchronize()`` or the timing measures kernel-queueing
      rather than compute.
    * **peak_mem_mb** - peak activation plus parameter memory in a forward
      pass. Often the binding constraint on a microcontroller, and frequently
      the real reason a model cannot be deployed even when its parameter count
      looks acceptable.

    ``mean_epoch_time_s`` is copied from ``history.json``, never re-measured -
    Member 3's trainer is the only thing that times epochs.
    """
    raise NotImplementedError("Member 1: implement profile_model")


def count_macs(model: nn.Module, input_shape: tuple[int, int, int]) -> int:
    """MACs for a single forward pass, via ``thop.profile``.

    ``thop`` mutates the model by attaching ``total_ops`` buffers. Profile a
    ``copy.deepcopy`` of the model, or those buffers end up in the checkpoint
    and in ``model_size_kb``, inflating the reported size.
    """
    raise NotImplementedError("Member 1: implement count_macs")


def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, int, int],
    device: str = "cpu",
    batch_size: int = 1,
    warmup: int = 10,
    repeats: int = 100,
) -> float:
    """Mean single-sample inference latency in milliseconds.

    Warm-up matters: the first few forward passes pay lazy CUDA context setup,
    cuDNN algorithm selection and allocator warm-up, and including them can
    inflate the mean several-fold.

    Report the median alongside the mean if the variance is high - a laptop
    under thermal throttling produces a long tail that makes the mean
    unrepresentative.
    """
    raise NotImplementedError("Member 1: implement measure_latency")


def measure_model_size_kb(model: nn.Module) -> float:
    """Serialised ``state_dict`` size in kilobytes.

    Save to a temporary file with ``torch.save`` and measure it. Always return
    KB; ``resources.schema.json`` fixes the unit so a KB/MB mix-up cannot
    reach the Section 6 table, where it would make the argument wrong rather
    than merely imprecise.
    """
    raise NotImplementedError("Member 1: implement measure_model_size_kb")
