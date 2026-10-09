"""Evaluation, benchmarking and report assembly.

Ownership is split by marking section, so each member defends their own marks:

    metrics.py    Member 1  - accuracy, precision, recall, confusion matrix
                              (inline in notebooks 03, 04, 05)
    benchmark.py  Member 1  - params, size, MACs, latency, peak memory (notebook 06)
    plotting.py   Member 1  - shared figure style only
    curves.py     Member 3  - Section 4 loss curves, Section 3 overlay
    reporting.py  Member 4  - Sections 3/4/6 tables and comparison figures (notebook 07)

Members 1, 3 and 4 all write here, but never to the same file. Every function
returns its result in every MODE; files are written only for official runs.
"""

from edgecnn.evaluation.benchmark import (
    count_macs,
    measure_latency,
    measure_model_size_kb,
    measure_peak_memory_mb,
    profile_all,
    profile_model,
)
from edgecnn.evaluation.metrics import compute_metrics, evaluate_run, predict

__all__ = [
    "compute_metrics",
    "count_macs",
    "evaluate_run",
    "measure_latency",
    "measure_model_size_kb",
    "measure_peak_memory_mb",
    "predict",
    "profile_all",
    "profile_model",
]
