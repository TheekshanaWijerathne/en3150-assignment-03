# EN3150 Assignment 03: CNNs for land-cover classification on small devices

**Group:** Misclassified

This folder holds the code of the assignment as seven Jupyter notebooks, together with the results
they produced. Every notebook contains all the code it runs, with comments; nothing is imported
from outside the notebook except standard Python packages.

The study classifies EuroSAT satellite images (64×64 pixels, 10 land-cover classes) with two custom
CNNs and two pretrained lightweight models, and compares their accuracy against what they cost to
store and to run.

## The notebooks

| Notebook | Assignment part | What it does |
|---|---|---|
| `01_data_preparation.ipynb` | Dataset | Downloads EuroSAT, draws the 70 / 15 / 15 split, shows the class distribution and sample images |
| `02_custom_architectures.ipynb` | Part 2 | Defines Model A and Model B, derives every layer's parameters, counts multiply-accumulates |
| `03_optimizer_study.ipynb` | Part 3 | Trains Model B with SGD, SGD with momentum and Adam, and compares them |
| `04_custom_training_evaluation.ipynb` | Part 4 | Trains Model A, evaluates it, and sets it against Model B |
| `05_pretrained_finetuning.ipynb` | Part 5 | Fine-tunes MobileNetV2 and SqueezeNet 1.1 on the same split |
| `06_resource_benchmark.ipynb` | Parts 4 and 6 | Measures size, MACs, CPU latency and peak memory of all four models |
| `07_final_comparison.ipynb` | Parts 3, 4 and 6 | Builds the comparison tables and the accuracy-versus-cost figures |

## How to run

Run the notebooks **in order, from this folder**, each from top to bottom. They pass results to each
other through the `results/` folder, so they must share it:

1. `01` and `02` need no GPU and take a few minutes.
2. `03`, `04` and `05` train models. Use a GPU: on an NVIDIA T4 they take about 20, 6 and 12
   minutes; on a CPU, hours. Each of them prepares the data itself, so it can also run on its own;
   `04` shows Model B's results from `03` when they are present.
3. `06` measures latency on one CPU thread, so run it on a computer that is otherwise idle. It takes
   about a minute.
4. `07` only reads the result files and takes seconds.

`results/` already holds the results of our runs, the numbers in the report, so `06` and `07` can
also run without any training. Running a notebook overwrites its own result files.

Everything a notebook downloads or writes stays inside the folder it runs from: `data/`,
`checkpoints/` and `results/`.

### Locally (Jupyter or VS Code)

Python 3.10 or newer, and the packages in `requirements.txt` (`pip install -r requirements.txt`).
With an NVIDIA GPU, install the CUDA build of PyTorch from <https://pytorch.org> first. Unzip the
folder, open the notebooks from it, and run them in order.

### On Google Colab

Colab already provides every package needed. Two ways:

**All notebooks in one session** (they share their files, as when run locally):

1. Open a new notebook, choose **Runtime > Change runtime type > T4 GPU**, and upload
   `Misclassified_A03_EN3150.zip` in the **Files** panel on the left.
2. Run, in one cell:

   ```
   !unzip -q Misclassified_A03_EN3150.zip
   %cd Misclassified_A03_EN3150
   ```

3. Run the notebooks **from this same notebook**, in order, one per cell:
   `%run 01_data_preparation.ipynb`, then `%run 02_custom_architectures.ipynb`, and so on to
   `%run 07_final_comparison.ipynb`. Printed results and figures appear under each cell; the full
   tables are written to `results/`.

Do not double-click the notebooks in the **Files** panel: Colab shows a file from there only as raw
text. To read a notebook with its saved outputs, open it on GitHub (see below), or in Colab with
**File > Open notebook > GitHub**.

**From Google Drive:** a notebook opened from Drive still runs in Colab's temporary `/content`
folder, not in its Drive folder, so its files would not reach Drive, and unpacking the dataset
(27,000 files) inside Drive would be slow. Copy the folder to the Colab machine instead, run it
there, and copy the results back. In a new GPU notebook, change the Drive path below to where the
folder is, then run:

```
from google.colab import drive
drive.mount("/content/drive")
!cp -r "/content/drive/MyDrive/Misclassified_A03_EN3150" /content
%cd /content/Misclassified_A03_EN3150
```

Then `%run` the notebooks in order as above. To keep the new results in Drive afterwards:
`!cp -r results "/content/drive/MyDrive/Misclassified_A03_EN3150"`.

**One notebook at a time:** upload a single notebook with **File > Upload notebook**, choose a GPU
runtime and run all. Every notebook runs this way except `07`, which needs the result files of `03`
to `06`, and `04` then skips its comparison with Model B. Each Colab notebook runs on its own
machine, so files written by one are not seen by the next.

No account, key or secret is needed: the notebooks download only public data (EuroSAT) and public
ImageNet weights.

### Downloads and disk space

EuroSAT through torchvision (about 94 MB, into `data/`), a cache of the decoded images (about
330 MB, also in `data/`), and the ImageNet weights of MobileNetV2 and SqueezeNet 1.1 (about 14 MB
and 5 MB, into `data/torch_hub/`). Trained weights are saved in `checkpoints/`.

## Results

| Path | Contents |
|---|---|
| `results/dataset.json` | Split sizes, the split fingerprint, the normalisation statistics |
| `results/<run>/history.json` | Loss, accuracy, learning rate and time of every epoch of one training run |
| `results/<run>/test_metrics.json` | Test accuracy, precision, recall, F1, per-class scores, confusion matrix |
| `results/<run>/curves.png`, `confusion_matrix.png` | The run's loss curves and confusion matrix |
| `results/benchmark.json` | Size, MACs, latency and peak memory of each model |
| `results/tables/` | The comparison tables, in Markdown |
| `results/figures/` | The dataset figures, the optimizer comparison and the accuracy-versus-cost figures |

The six runs are `model_b_sgd`, `model_b_sgd_momentum`, `model_b_adam`, `model_a_adam`,
`mobilenet_v2_adam` and `squeezenet1_1_adam`.

## Reproducibility

- Every random choice is seeded with 42: the split, the initial weights, augmentation and batch
  order. PyTorch is asked for deterministic algorithms.
- The split is drawn with NumPy's legacy `RandomState`, whose sequence does not change between
  versions. Its SHA-256 fingerprint is checked against the split of the reported results before any
  training starts.
- On an NVIDIA T4 with the versions below, a second complete run reproduced every accuracy, loss
  and confusion matrix exactly; only the timings differed. Another GPU or other versions may change
  the last digits.
- The reported numbers come from these machines:
  - training (`03`, `04`, `05`): {{training_environment}}
  - latency (`06`): {{benchmark_environment}}

## Source repository

The development repository, with the complete history and the automated tests, is at
<{{repository}}>.

## References

- P. Helber, B. Bischke, A. Dengel and D. Borth, "EuroSAT: A Novel Dataset and Deep Learning
  Benchmark for Land Use and Land Cover Classification", IEEE Journal of Selected Topics in Applied
  Earth Observations and Remote Sensing, 2019.
- M. Sandler, A. Howard, M. Zhu, A. Zhmoginov and L.-C. Chen, "MobileNetV2: Inverted Residuals and
  Linear Bottlenecks", CVPR 2018.
- F. N. Iandola, S. Han, M. W. Moskewicz, K. Ashraf, W. J. Dally and K. Keutzer, "SqueezeNet:
  AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size", arXiv:1602.07360, 2016.
- PyTorch and torchvision, <https://pytorch.org>.
