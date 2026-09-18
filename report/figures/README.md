# `report/figures/`

**Do not put anything here by hand.**

`scripts/build_report_assets.py` mirrors `results/figures/` into this folder. One source of truth
means a figure in the report can never drift out of sync with the numbers in the tables beside it.

```powershell
python scripts/build_report_assets.py
```

If a figure is missing, the run that produces it has not been committed yet — check
`results/metrics/` for the corresponding `<run_id>`.
