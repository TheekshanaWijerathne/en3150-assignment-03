# `data/` — SEAM 1 producer

**Owner: Member 1.** Consumed by Members 2, 3 and 4. Assignment §1.

Everything about the dataset lives here, and nothing about it lives anywhere else. No other member
opens an image file, computes a normalisation statistic, or decides which sample is in which split.

## Inputs

| From | What |
|---|---|
| `configs/stages/data.yaml` | dataset name and source, image size, split fractions, augmentation, seed |
| EuroSAT via torchvision | 27,000 satellite images, 10 classes, natively 64×64 RGB |

## Outputs

| Path | Committed? | Consumed by |
|---|---|---|
| `data/processed/eurosat_64/` | no — regenerable | this package |
| `data/splits/split_manifest.csv` | **yes** | everyone |
| `data/splits/split_meta.json` | **yes** | everyone, and the report |
| `data/splits/norm_stats.json` | **yes** | this package |
| `DataBundle` (in memory) | — | Members 2, 3, 4 |

## The contract you are providing

```python
data = build_dataloaders(cfg)     # -> DataBundle

images : float32, (B, 3, 64, 64), normalised with TRAIN-ONLY statistics
labels : int64,   (B,),           values in [0, num_classes)
```

`split_manifest.csv` columns, in this order:
`relative_path, label_index, label_name, split`

## Files

| File | Purpose |
|---|---|
| `synthetic.py` | **Build this first.** The Phase 0 unblocker — a tiny generated dataset satisfying the full contract, so Members 3 and 4 can start before EuroSAT downloads |
| `prepare.py` | Download, resize, stratified split, write the three committed artifacts |
| `loaders.py` | `build_dataloaders(cfg) -> DataBundle` |
| `transforms.py` | Augmentation and normalisation pipelines |

## Order of work

1. `make_synthetic_fixture` — **three other people are waiting on this.**
2. `build_dataloaders` with the `synthetic` branch working end to end.
3. `prepare_dataset` for the real EuroSAT download.
4. Commit the three split files. Announce it — that is the moment everyone switches off synthetic.

## Things that silently corrupt results

- Augmenting val or test. Train only.
- Fitting normalisation on anything but train. The schema pins `fitted_on: "train"`.
- Backslashes in `relative_path`. The manifest is committed and the team runs Windows and Linux.
- Redrawing the split after results exist. `split_meta.json` carries the manifest's SHA-256 so this
  is detectable; `--force` is required to overwrite.
- An unstratified split. A class landing entirely in train leaves its test recall undefined, which
  breaks the macro average.
