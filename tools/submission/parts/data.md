## Data

EuroSAT (Helber et al., 2019): 27,000 Sentinel-2 satellite images of 64×64 pixels, in 10 land-cover
classes. It is downloaded through torchvision and split 70 / 15 / 15 into training, validation and
test sets, stratified by class.

The split is drawn from a fixed seed with NumPy's legacy `RandomState`, whose random sequence does not
change between NumPy versions, so every machine draws the same split. A SHA-256 fingerprint of the
split list (each image's path, label and split) is checked against the fingerprint of the split used
for the reported results: if the dataset or the code differed, the check stops the notebook.

```python
import csv
import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

SPLIT_NAMES = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_SIZE = (64, 64)
EXPECTED_IMAGES = 27_000
BATCH_SIZE = 64

# SHA-256 of the split behind every reported result (see split_fingerprint below).
EXPECTED_SPLIT_SHA256 = "9fc7491cc0b8266013ca7f91e4a0988f39464988545251ee1361552823ea162b"


def download_eurosat():
    """Download EuroSAT through torchvision (about 94 MB, only once) and list its images.

    Returns:
        The class names, in label order (torchvision sorts the folder names), and
        one row per image, sorted by its path inside the dataset, such as
        ``AnnualCrop/AnnualCrop_1.jpg``.
    """
    dataset = torchvision.datasets.EuroSAT(root=str(DATA_DIR), download=True)
    classes = list(dataset.classes)
    rows = [
        {
            "relative_path": f"{classes[label]}/{Path(path).name}",
            "label_index": int(label),
            "label_name": classes[label],
            "source": Path(path),
        }
        for path, label in dataset.samples
    ]
    rows.sort(key=lambda row: row["relative_path"])
    if len(classes) != NUM_CLASSES or len(rows) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"found {len(classes)} classes and {len(rows)} images, expected {NUM_CLASSES} and "
            f"{EXPECTED_IMAGES}: the download is incomplete. Delete {DATA_DIR / 'eurosat'} and run again."
        )
    return classes, rows


def split_sizes(n, fractions):
    """Train, validation and test sizes for one class of ``n`` images."""
    sizes = [round(n * fractions["train"]), round(n * fractions["val"])]
    sizes.append(n - sum(sizes))
    if n >= 3:  # move images from train so that no split is empty
        for index in (1, 2):
            if sizes[index] < 1:
                sizes[0] -= 1 - sizes[index]
                sizes[index] = 1
    return sizes


def stratified_split(labels, fractions, seed):
    """Assign every image to ``train``, ``val`` or ``test``, class by class.

    Within each class the images are shuffled, then cut in the given proportions.
    Splitting per class keeps the class balance identical in all three splits.

    Returns:
        The split name of each image, in the order of ``labels``.
    """
    labels = np.asarray(labels)
    assignment = np.empty(len(labels), dtype=object)
    rng = np.random.RandomState(seed)
    for label in np.unique(labels):  # sorted, so the random sequence is fixed
        members = rng.permutation(np.flatnonzero(labels == label))
        start = 0
        for name, count in zip(SPLIT_NAMES, split_sizes(len(members), fractions)):
            assignment[members[start : start + count]] = name
            start += count
    return assignment.tolist()


def split_fingerprint(rows):
    """SHA-256 of the split list, written as a CSV of path, label index, label name and split."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["relative_path", "label_index", "label_name", "split"])
    for row in rows:
        writer.writerow([row["relative_path"], row["label_index"], row["label_name"], row["split"]])
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def load_pixels(rows, cache_path):
    """Every image as one ``(N, 64, 64, 3)`` ``uint8`` array, in row order.

    Decoding a JPEG is a fixed step with nothing to learn, so it is done once and
    the pixels are cached next to the dataset. Augmentation and normalisation are
    still applied fresh in every epoch, on top of these pixels.
    """
    height, width = IMAGE_SIZE
    if cache_path.exists():
        pixels = np.load(cache_path)
        if pixels.shape == (len(rows), height, width, 3):
            return pixels

    pixels = np.empty((len(rows), height, width, 3), dtype=np.uint8)

    def decode(position):
        with Image.open(rows[position]["source"]) as image:
            array = np.asarray(image.convert("RGB"))
        if array.shape != (height, width, 3):
            raise ValueError(f"{rows[position]['relative_path']} is not {width}x{height}")
        pixels[position] = array

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        list(pool.map(decode, range(len(rows))))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, pixels)
    return pixels


def channel_stats(pixels, indices, chunk=1024):
    """Per-channel mean and standard deviation of the selected images, on a 0 to 1 scale.

    Computed in chunks with 64-bit sums, so the whole training set never needs
    more than a few tens of MB of extra memory.
    """
    indices = np.asarray(indices)
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    for start in range(0, len(indices), chunk):
        block = pixels[indices[start : start + chunk]].astype(np.float64) / 255.0
        total += block.sum(axis=(0, 1, 2))
        total_sq += np.square(block).sum(axis=(0, 1, 2))
    count = len(indices) * pixels.shape[1] * pixels.shape[2]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - np.square(mean), 0.0))
    return mean.tolist(), std.tolist()


@dataclass
class EuroSATData:
    """The prepared dataset: images, labels, the split and the normalisation statistics."""

    class_names: list
    rows: list
    labels: np.ndarray
    splits: dict          # split name -> indices into ``pixels``
    pixels: np.ndarray
    norm_mean: list       # of the training images only
    norm_std: list
    fingerprint: str


def prepare_data():
    """Download EuroSAT, draw the split, check its fingerprint and decode the images.

    The normalisation statistics come from the training images only: including
    validation or test images would leak held-out information into training.
    """
    classes, rows = download_eurosat()
    labels = [row["label_index"] for row in rows]
    for row, split in zip(rows, stratified_split(labels, SPLIT_FRACTIONS, SEED)):
        row["split"] = split

    fingerprint = split_fingerprint(rows)
    if fingerprint != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(
            f"the split's fingerprint is {fingerprint[:16]}, not {EXPECTED_SPLIT_SHA256[:16]}: "
            "this is not the split behind the reported results"
        )

    pixels = load_pixels(rows, DATA_DIR / f"eurosat_pixels_{fingerprint[:16]}.npy")
    split_of = np.array([row["split"] for row in rows])
    splits = {name: np.flatnonzero(split_of == name) for name in SPLIT_NAMES}
    mean, std = channel_stats(pixels, splits["train"])
    return EuroSATData(
        class_names=classes,
        rows=rows,
        labels=np.array(labels, dtype=np.int64),
        splits=splits,
        pixels=pixels,
        norm_mean=mean,
        norm_std=std,
        fingerprint=fingerprint,
    )
```

Training images get random flips and quarter turns in every epoch. Satellite images have no natural
"up", so a flipped or rotated image is still a valid example. Validation and test images are never
augmented, so they are identical every time a model is evaluated.

```python
class RandomRot90:
    """Rotate an image by 0, 90, 180 or 270 degrees, chosen at random.

    A quarter turn of a square image loses no pixels, unlike an arbitrary angle.
    """

    def __call__(self, image):
        quarter_turns = int(torch.randint(0, 4, (1,)).item())
        return torch.rot90(image, quarter_turns, dims=(-2, -1))

    def __repr__(self):
        return "RandomRot90()"


def build_transform(split, mean, std):
    """Augmentation for the training split only, then scaling to 0 to 1 and normalisation."""
    steps = []
    if split == "train":
        steps += [v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5), RandomRot90()]
    steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=list(mean), std=list(std))]
    return v2.Compose(steps)


class PixelDataset(Dataset):
    """Images from the decoded-pixel array, returned as ``(normalised tensor, label)``."""

    def __init__(self, pixels, indices, labels, transform):
        self.pixels = pixels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image = torch.from_numpy(self.pixels[self.indices[item]]).permute(2, 0, 1)  # HWC to CHW
        return self.transform(image), int(self.labels[item])


def make_loaders(data, mean, std):
    """Training, validation and test DataLoaders, normalised with ``mean`` and ``std``.

    The training loader reshuffles in every epoch, with its own generator seeded
    from ``SEED``; validation and test keep a fixed order.
    """
    loaders = {}
    for split in SPLIT_NAMES:
        indices = data.splits[split]
        dataset = PixelDataset(
            data.pixels, indices, data.labels[indices], build_transform(split, mean, std)
        )
        is_train = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=is_train,
            generator=torch.Generator().manual_seed(SEED) if is_train else None,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    return loaders
```
