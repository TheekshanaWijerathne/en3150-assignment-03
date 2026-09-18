# `data/`

**Owner: Member 1.** Nobody else writes here.

| Folder | Contents | Committed? |
|---|---|---|
| `raw/` | Original EuroSAT download | **no** — large, and reproducible from `configs/stages/data.yaml` |
| `processed/` | Resized 64×64 images | **no** — a deterministic transform of `raw/` |
| `splits/` | The split definition | **yes** — this is the reproducibility contract |

## Why `splits/` is committed and the images are not

`split_manifest.csv` is what guarantees Model A, Model B, MobileNetV2 and SqueezeNet saw
byte-identical data. An accuracy difference between them is then attributable to the architecture,
not to a lucky split. It is a small text file, so it costs nothing to commit and everything to
lose.

The images are a download plus a deterministic resize. Anyone can regenerate them:

```powershell
python scripts/prepare_data.py --config configs/stages/data.yaml
```

## `splits/` contents

| File | Purpose |
|---|---|
| `split_manifest.csv` | One row per image: `relative_path, label_index, label_name, split` |
| `split_meta.json` | Class names (frozen ordering), per-split counts, seed, and the manifest's SHA-256 |
| `norm_stats.json` | Per-channel mean and std, fitted on the **train split only** |

## Rules

1. **The split is drawn once.** `prepare_data.py` requires `--force` to overwrite it, and doing so
   invalidates every result already in `results/metrics/`.
2. **`split_meta.json` carries the manifest's SHA-256.** If it stops matching, someone regenerated
   or hand-edited the split and the committed results are stale.
3. **`class_names` ordering is frozen** once committed. Every confusion matrix, per-class metric
   and figure legend is indexed by it.
4. **Normalisation is fitted on train only.** The schema pins `fitted_on: "train"` so the mistake
   cannot be committed silently.
5. **Forward slashes in `relative_path`.** The manifest is committed and the team runs Windows and
   Linux; `schema.validate_manifest` rejects backslashes.

## After Member 1 commits these

Announce it. That is the moment everyone else switches off `dataset.name: synthetic` and onto the
real data.
