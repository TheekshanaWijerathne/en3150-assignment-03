# `evaluation/` — SEAM 4 and SEAM 5

**Three owners, no shared files.** Ownership follows the marking sections, so each member defends
their own marks.

| File | Owner | Section | Where it runs |
|---|---|---|---|
| `metrics.py` | Member 1 | §4 — accuracy, precision, recall, confusion matrix | inline in `03`, `04`, `05`, right after training |
| `benchmark.py` | Member 1 | §4, §6 — params, size, MACs, latency, peak memory | notebook `06`, one session on one CPU |
| `plotting.py` | Member 1 | shared figure style only | every notebook's style cell |
| `curves.py` | Member 3 | §4 loss curves, §3 optimizer overlay | `03`, `04`, `05` |
| `reporting.py` | Member 4 | §3/§4/§6 tables, confusion matrices, trade-off plots | `04`, `05`, `07` |

## Every function returns, and writes only when told to

- Functions for a single run follow `cfg.is_official`: `evaluate_run` writes `test_metrics.json` only
  for an official run, and always returns an `EvalResult`.
- Plot and table functions take an explicit `write=`. Notebooks pass `write=cfg.is_official` for a
  single run, or `write = (MODE == "official")` in `07`. Every one **returns** its Figure or
  Markdown, so a debug run shows everything inline.
- Plot functions accept either a `run_id` (reading committed JSON) or a result still in memory.

## Seam 4 — Member 1 produces, Member 4 consumes

**Inputs:** the best checkpoint (`TrainResult.checkpoint_path`), `DataBundle.test`, and each run's
experiment config and `history.json`.
**Outputs (official runs, committed):**

- `results/metrics/<run_id>/test_metrics.json` — from `evaluate_run`, in `03`–`05`;
- `results/metrics/<run_id>/resources.json` — from `profile_all`, in `06`.

## Seam 5 — Member 4 produces

**Inputs:** the committed JSON under `results/metrics/`.
**Outputs:** `results/tables/*.md` and `results/figures/*.png`, mirrored into `report/figures/`.

## Why resources are measured separately, in one session

Latency and peak memory depend on the machine, but **not on trained weights**: an architecture costs
the same with random weights as with trained ones. So notebook `06` builds all four architectures
with `build_model_from_config` and profiles them one after another, on one CPU. It needs no
checkpoints, and every row of the §6 table is measured on the same hardware, whoever trained which
model where. Test accuracy does not depend on hardware, so it is computed inline, straight after
training.

Epoch time is the one hardware-dependent number that comes from training. `check_same_hardware` in
`07` warns when a table mixes devices.

## Why one owner for the metrics

Four implementations of "precision" differ in averaging, zero-division handling and class ordering.
The §6 table would then compare numbers that are not comparable, and the error would be invisible,
because every column still looks plausible. One code path removes that risk.

## Fixed conventions

- **Confusion matrix orientation:** `cm[i][j]` = true class `i` predicted as `j`. Stored as raw
  counts; normalised at plot time only, by **true class**, so the diagonal reads as per-class recall.
- **Macro averaging.** For single-label classification, micro-precision equals accuracy, so a micro
  column would silently repeat the accuracy column.
- **Latency on CPU, one thread, batch size 1, for every model** — the §6 argument is about edge
  devices. The mean of 100 timed passes after 10 untimed ones; the median is recorded too.
- **Peak memory** = the weights plus the largest set of intermediate results alive at once, worked
  out from the model's graph (`torch.fx`): each result is freed after its last use, and views and
  in-place results are counted once. Machine-independent.
- **Units.** `model_size_kb` is always KB; convert to MB only for the §5 table. 1 KB = 1,024 bytes
  and 1 MB = 1,048,576 bytes, as in torchvision's published file sizes.
- **Figure resolution.** Inline figures at `INLINE_DPI` keep committed notebooks small; saved PNGs
  use `FIGURE_DPI`, for print.

## Traps

- `count_macs` counts **convolution and fully-connected** multiply-accumulates only, the convention
  of the MobileNet and SqueezeNet papers; it reproduces torchvision's published figures exactly.
  Many papers report FLOPs instead, roughly `2 × MACs`. Say which you used.
- Measure `model_size_kb` by serialising the `state_dict`, not as `numel * 4` and not from the
  checkpoint file. The computed figure misses buffers such as BatchNorm running statistics, which
  take real bytes on the device; a checkpoint also holds optimizer state, several times the weights.
- Pass `labels=range(len(class_names))` to `sklearn.confusion_matrix`. Otherwise a class absent from
  the predictions disappears and the matrix stops being square.

## Sanity check worth asserting

The confusion matrix must sum to `split_meta.json → counts.test`. If it does not, the wrong split was
evaluated.
