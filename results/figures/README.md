# `results/figures/` — generated figures

**COMMITTED.** Producers: Member 3 (curves), Member 4 (everything else).

```
results/figures/<run_id>/curves.png              Section 4 — train/val loss and accuracy
results/figures/<run_id>/confusion_matrix.png    Section 4 — per-run confusion matrix
results/figures/optimizer_overlay.png            Section 3 — sgd vs sgd_momentum vs adam
results/figures/accuracy_vs_params.png           Section 6 — the trade-off
results/figures/accuracy_vs_macs.png             Section 6 — compute, not just memory
```

Figures are **generated, never hand-made**. `scripts/build_report_assets.py` mirrors this folder
into `report/figures/`, so there is exactly one source of truth and figures cannot drift out of
sync with the numbers in the tables.

Style comes from `edgecnn.evaluation.plotting` — shared colours per model and per optimizer, so a
reader who learns "orange is Model B" on the loss curves does not relearn it on the scatter plot.

Confusion matrices are stored as raw counts in JSON and normalised at plot time, by **true class**,
so the diagonal reads as per-class recall.
