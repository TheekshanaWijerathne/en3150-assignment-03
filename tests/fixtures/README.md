# `tests/fixtures/` — the Phase 0 unblocker

**Owner: Member 1. Build this before anything else in the data stage.**

## Why this is the first thing Member 1 writes

The pipeline split puts Members 3 and 4 downstream of Members 1 and 2. Without a fixture they wait
— for a dataset download, then for a real loader, then for real models. That is three days of two
people idle.

With it, they do not wait at all. The fixture satisfies the **full Seam 1 contract** — same shapes,
same dtypes, same manifest columns, same `DataBundle` — using generated noise instead of satellite
imagery. Member 3 can run a complete training loop, write a schema-valid `history.json` and debug
checkpointing on hour one.

## What `make_synthetic_fixture` produces

```
tests/fixtures/synthetic/
├── images/<class_name>/*.png     ~200 generated 64x64 RGB images, 4 classes
├── split_manifest.csv            real manifest, real columns, 70/15/15
├── split_meta.json               real metadata
└── norm_stats.json               real train-only statistics
```

Deterministic given a seed, so two members debugging the same failure see the same batches. Small
enough that a full 20-epoch run finishes in seconds on a CPU.

## Using it

Set in any config:

```yaml
inputs:
  dataset:
    name: synthetic
```

Or from the CLI:

```powershell
python scripts/prepare_data.py --config configs/stages/data.yaml --synthetic
```

## Requirements

- **Give each class a distinguishable statistical signature** — a different per-channel mean plus
  noise is enough. A model that cannot fit this is genuinely broken, which makes the fixture a real
  smoke test rather than only a shape check.
- **Validate its own outputs** against `configs/contracts/*.schema.json` before returning. If the
  fixture does not satisfy the contract, it is not a fixture.
- **Fully deterministic** given `seed`.

## One rule

**Results produced from synthetic data must never be committed to `results/metrics/`.** The class
structure is fabricated, so the accuracy is meaningless. `scripts/train.py` refuses when
`subset_fraction < 1.0`, but the synthetic dataset is a separate trap — check the dataset name
before committing a metrics file.
