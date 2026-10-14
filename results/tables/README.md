# `results/tables/` — comparison tables

**COMMITTED.** Markdown, so the tables paste straight into the report. Written only by official
runs.

| File | Section | Notebook | Owner |
|---|---|---|---|
| `custom_architectures.md` | 2 | `02` | Member 2 — layer-by-layer topology |
| `param_breakdown.md` | 2 | `02` | Member 2 — per-layer parameter derivation |
| `optimizer_comparison.md` | 3 | `07` | Member 4 — SGD vs SGD+momentum vs Adam |
| `custom_model_comparison.md` | 4 | `07` | Member 4 — Model A vs Model B |
| `final_comparison.md` | 6 | `07` | Member 4 — Model B vs the SOTA backbones |

Notebook `07` builds its tables only from committed JSON. Every table is also returned and shown in
the notebook, so a debug pass previews them without writing anything.

## What each must contain

**Section 4** asks for: total parameter count, estimated model size on disk (KB), training time per
epoch, and test accuracy, precision and recall. Add MACs — that is the number that actually shows
the saving from depthwise-separable convolutions.

**Section 3** asks about both convergence *and* final performance, so report both: `best_val_acc` for
performance, and the epochs each run needed to reach 90% of its own best validation accuracy for
speed. The two can disagree, and that is worth discussing.

**Section 6** is the central table: trainable params, total params, size, MACs, CPU inference
latency, peak memory, test accuracy, macro precision and recall. Give ratios, not just the two
numbers: "40x fewer parameters for 6 points of accuracy" is an argument; two columns of figures is
not.

Generated from committed JSON. Do not hand-edit.
