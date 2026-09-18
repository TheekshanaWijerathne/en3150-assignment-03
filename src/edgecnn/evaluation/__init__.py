"""Evaluation, benchmarking and report assembly.

Ownership is split by marking section, so each member defends their own marks:

    metrics.py    Member 1  - accuracy, precision, recall, confusion matrix
    benchmark.py  Member 1  - params, size, MACs, latency, peak memory
    plotting.py   Member 1  - shared figure style only
    curves.py     Member 3  - Section 4 loss curves, Section 3 overlay
    reporting.py  Member 4  - Sections 4/5/6 tables and comparison figures

Members 1 and 4 both write here, but never to the same file.
"""

from edgecnn.evaluation.benchmark import (
    count_macs,
    measure_latency,
    measure_model_size_kb,
    profile_model,
)
from edgecnn.evaluation.metrics import compute_metrics, evaluate_run, predict

__all__ = [
    "compute_metrics",
    "count_macs",
    "evaluate_run",
    "measure_latency",
    "measure_model_size_kb",
    "predict",
    "profile_model",
]
