# `data/raw/` — original dataset

**Git-ignored** (except this file). Owner: Member 1.

EuroSAT lands here via `torchvision.datasets.EuroSAT(root="data/raw", download=True)` —
27,000 satellite images, 10 classes, natively 64x64 RGB.

```powershell
python scripts/prepare_data.py --config configs/stages/data.yaml
```

Never edit anything in here by hand. This folder is a pure download; `data/processed/` is a
deterministic transform of it; `data/splits/` is the committed output that everyone consumes.
