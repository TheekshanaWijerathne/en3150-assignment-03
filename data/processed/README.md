# `data/processed/` — resized images

**Git-ignored** (except this file). Owner: Member 1.

Output of `scripts/prepare_data.py` — images at the final 64x64 resolution, laid out as
`eurosat_64/<class_name>/<image>.jpg`.

EuroSAT is already 64x64, so with `resize_mode: none` this is effectively a copy. The step exists
so the pipeline still works if the team ever changes dataset.

`relative_path` in `data/splits/split_manifest.csv` is relative to the folder in here named by
`outputs.processed_dir`.
