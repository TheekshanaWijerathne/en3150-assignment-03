# `evaluation/` — SEAM 4 and SEAM 5

**Two owners, no shared files.** Ownership follows the marking sections, so each member defends
their own marks.

| File | Owner | Section |
|---|---|---|
| `metrics.py` | Member 1 | §4 — accuracy, precision, recall, confusion matrix |
| `benchmark.py` | Member 1 | §4, §6 — params, size, MACs, latency, peak memory |
| `plotting.py` | Member 1 | shared figure style only |
| `curves.py` | Member 3 | §4 loss curves, §3 optimizer overlay |
| `reporting.py` | Member 4 | §4, §5, §6 tables and comparison figures |

## Seam 4 — Member 1 produces, Member 4 consumes

**Inputs:** `artifacts/checkpoints/<run_id>/best.pt`, `DataBundle.test`,
`results/metrics/<run_id>/history.json`

**Outputs (both committed):**

- `results/metrics/<run_id>/test_metrics.json`
- `results/metrics/<run_id>/resources.json`

## Seam 5 — Member 4 produces

**Inputs:** every committed JSON under `results/metrics/`
**Outputs:** `results/tables/*.md`, `results/figures/*.png`, mirrored into `report/figures/`

## Why one owner for the metrics

Four implementations of "precision" will differ in averaging, in zero-division handling, and in
class ordering. The §6 table would then compare numbers that are not comparable — and the error is
invisible, because every column still looks plausible. One code path removes the possibility.

The same applies to `benchmark.py`: every model is profiled by the same `profile_model` call, on
the same device, or the comparison is not a comparison.

## Fixed conventions

**Confusion matrix orientation.** `cm[i][j]` = true class `i` predicted as `j`. Rows sum to each
class's support. Stored as raw counts; normalise at plot time only, by **true class**, so the
diagonal reads as per-class recall.

**Macro averaging.** For single-label classification, micro-precision is numerically equal to
accuracy — a micro column would silently duplicate the accuracy column.

**Latency on CPU, batch size 1, for every model** — even on a CUDA machine. The §6 argument is
about edge deployment, and edge devices have no GPU. Mixing CPU and GPU timings makes the column
meaningless.

**Units.** `model_size_kb` is always KB. Convert to MB for the §5 table only. The schema fixes the
unit so a KB/MB mix-up cannot reach §6, where it would make the argument wrong rather than merely
imprecise.

## Traps

- `thop.profile` attaches `total_ops` buffers to the model. Profile a `deepcopy`, or those buffers
  end up in the checkpoint and inflate `model_size_kb`.
- Measure `model_size_kb` by serialising the `state_dict`, not as `numel * 4`. The computed figure
  misses buffers such as BatchNorm running statistics, which are real bytes on the device.
- Pass `labels=range(len(class_names))` to `sklearn.confusion_matrix`. Without it a class absent
  from the predictions vanishes and the matrix stops being square — which breaks the plot far from
  the cause.
- Cast NumPy scalars to plain `int`/`float` before serialising.
- thop reports **MACs**; many papers report FLOPs at roughly `2 x MACs`. State which you used, or
  the comparison against published MobileNet figures looks wrong by a factor of two.

## Sanity check worth asserting

The confusion matrix must sum to `split_meta.json -> counts.test`. If it does not, the wrong split
was evaluated — much better caught here than in the viva.
