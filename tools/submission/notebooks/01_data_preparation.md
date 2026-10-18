# 1. Data preparation

This notebook prepares the dataset used by every model in the study:

1. downloads EuroSAT through torchvision (about 94 MB, on the first run only);
2. draws the stratified 70 / 15 / 15 split and checks its fingerprint;
3. decodes every image once and computes the normalisation statistics of the training images;
4. shows the split sizes, the class distribution and sample images.

The training notebooks (03, 04 and 05) repeat steps 1 to 3 themselves, so each of them can also run
on its own.

**Writes:** `results/dataset.json`, `results/figures/class_distribution.png`,
`results/figures/sample_grid.png`.

<!-- include: common -->
<!-- include: data -->
<!-- include: dataset_figures -->

## Prepare the dataset

The first run downloads the dataset and decodes all 27,000 images, which takes a minute or two;
later runs reuse both.

```python
data = prepare_data()
print(f"{len(data.rows):,} images in {len(data.class_names)} classes: {', '.join(data.class_names)}")
print(f"Split fingerprint (SHA-256): {data.fingerprint}")
print("It matches the split behind the reported results.")
```

## Split sizes

The totals split 70 / 15 / 15, and every class is split in the same proportions.

```python
split_counts(data)
```

## Class distribution

```python
plot_class_distribution(data)
```

## Sample images

```python
plot_sample_grid(data)
```

## Normalisation statistics

Per-channel mean and standard deviation of the training images, on a 0 to 1 scale. The custom
models are normalised with these; the pretrained models with the ImageNet statistics their weights
were trained with.

```python
write_json(RESULTS_DIR / "dataset.json", {
    "num_images": len(data.rows),
    "class_names": data.class_names,
    "split_counts": {split: int(len(data.splits[split])) for split in SPLIT_NAMES},
    "split_sha256": data.fingerprint,
    "norm_mean": data.norm_mean,
    "norm_std": data.norm_std,
})
print("mean", [round(value, 4) for value in data.norm_mean])
print("std ", [round(value, 4) for value in data.norm_std])
```
