## Looking at the split

```python
import pandas as pd

SPLIT_COLORS = {"train": "#2F5E8E", "val": "#6C9BCF", "test": "#B3CDE8"}


def split_counts(data):
    """Images per class in each split, with totals."""
    table = pd.DataFrame(
        [[int(np.sum(data.labels[data.splits[split]] == label)) for split in SPLIT_NAMES]
         for label in range(len(data.class_names))],
        index=pd.Index(data.class_names, name="class"),
        columns=list(SPLIT_NAMES),
    )
    table.loc["Total"] = [len(data.splits[split]) for split in SPLIT_NAMES]
    table["total"] = table[list(SPLIT_NAMES)].sum(axis=1)
    return table.astype(int)


def plot_class_distribution(data):
    """Each class's share of every split; equal bars mean stratification kept the class balance."""
    counts = split_counts(data)
    positions = np.arange(len(data.class_names))
    width = 0.8 / len(SPLIT_NAMES)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(data.class_names) + 1.5), 4.2),
                           layout="constrained")
    tallest = 0.0
    for offset, split in enumerate(SPLIT_NAMES):
        total = int(counts.loc["Total", split])
        shares = [100 * int(counts.loc[name, split]) / total for name in data.class_names]
        tallest = max(tallest, *shares)
        ax.bar(positions + (offset - (len(SPLIT_NAMES) - 1) / 2) * width, shares, width,
               label=f"{split} ({total:,} images)", color=SPLIT_COLORS[split])
    ax.set_xticks(positions, data.class_names, rotation=35, ha="right")
    ax.set_ylabel("share of the split (%)")
    ax.set_ylim(0, tallest * 1.3)  # room for the legend above the bars
    ax.set_title("Class distribution in each split")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncols=len(SPLIT_NAMES))
    save_figure(fig, RESULTS_DIR / "figures" / "class_distribution.png")
    plt.show()


def plot_sample_grid(data, per_class=4):
    """Random training images, one row per class, unaugmented, as the models receive them."""
    rng = np.random.RandomState(SEED)
    names = data.class_names
    fig, axes = plt.subplots(len(names), per_class, squeeze=False,
                             figsize=(0.95 * per_class + 1.9, 0.95 * len(names) + 0.5),
                             layout="constrained")
    for label, name in enumerate(names):
        train = [position for position, row in enumerate(data.rows)
                 if row["split"] == "train" and row["label_index"] == label]
        picks = rng.choice(len(train), size=min(per_class, len(train)), replace=False)
        for column, ax in enumerate(axes[label]):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if column < len(picks):
                ax.imshow(data.pixels[train[picks[column]]], interpolation="nearest")
        axes[label][0].set_ylabel(name, rotation=0, ha="right", va="center", labelpad=6)
    fig.suptitle("Training images at 64×64, as the models receive them")
    save_figure(fig, RESULTS_DIR / "figures" / "sample_grid.png")
    plt.show()
```
