# 7. Final comparison

Builds the comparison tables of Sections 3, 4 and 6 and the accuracy-versus-cost figures, from the
result files of notebooks 03 to 06. Nothing is trained or measured here, so it runs in seconds.

**Writes:** `results/tables/optimizer_comparison.md`, `custom_model_comparison.md` and
`final_comparison.md`; `results/figures/accuracy_vs_params.png` and `accuracy_vs_macs.png`.

<!-- include: common -->
<!-- include: reporting -->

## The result files

```python
missing = [run for run in RUNS if not (RESULTS_DIR / run / "test_metrics.json").exists()]
if missing or not (RESULTS_DIR / "benchmark.json").exists():
    raise FileNotFoundError(f"run notebooks 03 to 06 first; missing: {missing or 'benchmark.json'}")
benchmark = read_json(RESULTS_DIR / "benchmark.json")

for run in RUNS:
    print(f"{run:<22} trained on {history_of(run)['device']}")
print(f"Latency measured on {benchmark['model_b']['device']}, one thread")
```

## Section 3: optimizers

```python
tables = {}
tables["optimizer_comparison"] = optimizer_table()
display(Markdown(tables["optimizer_comparison"]))
```

## Section 4: Model A against Model B

```python
tables["custom_model_comparison"] = custom_model_table(benchmark)
display(Markdown(tables["custom_model_comparison"]))
```

## Section 6: Model B against the pretrained models

```python
tables["final_comparison"] = final_table(benchmark)
display(Markdown(tables["final_comparison"]))
for name, text in tables.items():
    write_text(RESULTS_DIR / "tables" / f"{name}.md", text)
```

## Accuracy against cost

```python
plot_tradeoff(benchmark, "total_params")
```

```python
plot_tradeoff(benchmark, "macs")
```
