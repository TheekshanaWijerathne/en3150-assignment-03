# EN3150 Assignment 03 - Resource-Constrained CNN for Edge Image Classification

This repository is the project scaffold for **EN3150 Pattern Recognition - Assignment 03**. The assignment investigates the trade-off between classification accuracy, memory footprint, and computational cost for image classifiers intended for embedded or low-resource devices.

The repository currently contains only this README and empty, tracked directories. Implementation, experiment, and report files will be added through regular commits as the assignment progresses.

## Team

| Field | Details |
|---|---|
| Group number | To be added |
| Member names | To be added |
| Index numbers | To be added |
| Dataset | To be selected - CIFAR-10 is not permitted |
| Framework | To be selected - PyTorch or TensorFlow/Keras |

## Assignment requirements

The project will:

1. Select a low-resolution image-classification dataset from the UCI Machine Learning Repository or a similar source.
2. Resize images to a maximum resolution of $64 \times 64$ pixels.
3. Create fixed, reproducible training, validation, and test splits of 70%, 15%, and 15%.
4. Design and train two custom networks:
   - **Model A:** a standard CNN using 2D convolutions and max pooling.
   - **Model B:** a lightweight CNN using depthwise-separable convolutions, with no more than 100,000 trainable parameters.
5. Compare the selected optimizer with standard SGD and SGD with momentum.
6. Train both custom models for at least 20 epochs and evaluate them using accuracy, confusion matrix, precision, and recall.
7. Fine-tune two lightweight pretrained architectures, such as MobileNet, SqueezeNet, EfficientNet-B0, or ShuffleNet, using the same data splits.
8. Compare parameter count, model size, training time, test accuracy, memory footprint, and computational cost across the models.

## Repository structure

```text
en3150-assignment-03/
|-- README.md
|-- configs/                     # Dataset, model, optimizer, and experiment settings
|-- data/
|   |-- raw/                     # Original dataset files
|   |-- processed/               # Resized and preprocessed images
|   `-- splits/                  # Reproducible train/validation/test split definitions
|-- notebooks/                   # Exploration and analysis notebooks
|-- src/
|   |-- data/                    # Loading, preprocessing, and augmentation
|   |-- models/
|   |   |-- custom/              # Model A and Model B
|   |   `-- pretrained/          # Two lightweight fine-tuned models
|   |-- training/                # Training and optimizer-comparison logic
|   |-- evaluation/              # Metrics, confusion matrices, and benchmarking
|   `-- utils/                   # Shared utilities and reproducibility helpers
|-- scripts/                     # Command-line entry points for experiments
|-- artifacts/
|   |-- checkpoints/             # Saved training checkpoints
|   `-- exports/                 # Final serialized or edge-deployment models
|-- results/
|   |-- figures/                 # Loss curves and confusion matrices
|   |-- metrics/                 # Machine-readable evaluation results
|   `-- tables/                  # Model and optimizer comparison tables
|-- report/
|   `-- figures/                 # Figures prepared specifically for the report
`-- tests/                       # Automated checks for data and model code
```

## Planned experiment matrix

| Track | Model or optimizer | Required comparison |
|---|---|---|
| Custom architecture | Model A - standard CNN | Parameters, size, time per epoch, and test metrics |
| Custom architecture | Model B - depthwise-separable CNN | Same metrics; must remain at or below 100,000 trainable parameters |
| Optimizer baseline | SGD | Convergence and final performance |
| Optimizer baseline | SGD with momentum | Effect of momentum on convergence and performance |
| Selected optimizer | To be chosen | Justification, learning rate, convergence, and final performance |
| Pretrained model 1 | To be chosen | Parameters, model size, and test metrics |
| Pretrained model 2 | To be chosen | Parameters, model size, and test metrics |

All models will use the same dataset split to ensure a fair comparison. Random seeds, preprocessing, augmentation, stopping criteria, and hardware details will be recorded with each experiment.

## Reproducibility rules

- Keep the test set isolated until final evaluation.
- Fit preprocessing operations using training data only.
- Store the exact split definition and random seed under `data/splits/`.
- Record model hyperparameters and optimizer settings under `configs/`.
- Report trainable parameter counts, serialized model sizes, training time per epoch, and evaluation hardware.
- Commit work regularly so that the development history reflects progress over time.

## Deliverables

- Complete, documented, runnable source code.
- Trained and evaluated custom CNNs and lightweight pretrained models.
- Training and validation loss curves.
- Accuracy, confusion matrix, precision, and recall for each evaluated model.
- Comparison tables covering accuracy, parameter count, model size, training time, memory footprint, and computational cost.
- A report named according to the required format: `YourGroupNo_A03_EN3150`.
- The repository/profile link included in the report.

## Status

**Scaffold only.** No dataset, source code, model checkpoint, result, or report has been added yet.
