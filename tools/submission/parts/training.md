## Training

One training loop serves all four models, so their training times and accuracies are measured the
same way. Settings shared by every run:

- **30 epochs**, cross-entropy loss, batch size 64;
- **cosine annealing** of the learning rate, from the optimizer's rate down to 0.00001;
- **early stopping** on validation accuracy with a patience of 8 epochs, never before epoch 20;
- the **checkpoint with the best validation accuracy** is the one evaluated on the test set.

The test split is never used during training; model selection uses the validation split only.
Each epoch's time covers the training pass and the validation pass, and on a GPU the clock stops
only after all queued GPU work has finished.

```python
import time

from torch import nn

EPOCHS = 30
MIN_LR = 1e-5           # floor of the cosine learning-rate schedule
EARLY_STOP_PATIENCE = 8
MIN_EPOCHS = 20

OPTIMIZER_SETTINGS = {
    "sgd": {"lr": 0.01, "momentum": 0.0, "weight_decay": 1e-4},
    "sgd_momentum": {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4},
    "adam": {"lr": 0.001, "betas": (0.9, 0.999), "weight_decay": 1e-4},
}


def param_groups(model, base_lr, backbone_lr_scale):
    """Two optimizer groups for a pretrained model: its trainable feature blocks at a lower
    rate, its new classifier at ``base_lr``."""
    head = [p for p in model.classifier.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    return [
        {"params": backbone, "lr": base_lr * backbone_lr_scale, "name": "backbone"},
        {"params": head, "lr": base_lr, "name": "head"},
    ]


def build_optimizer(model, name, backbone_lr_scale=None):
    """The optimizer ``name`` over the model's trainable parameters.

    For a pretrained model, ``backbone_lr_scale`` puts the trainable feature
    blocks in their own group at a fraction of the learning rate.
    """
    settings = OPTIMIZER_SETTINGS[name]
    lr = settings["lr"]
    if backbone_lr_scale is None:
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = param_groups(model, lr, backbone_lr_scale)
    if name in ("sgd", "sgd_momentum"):
        return torch.optim.SGD(params, lr=lr, momentum=settings["momentum"],
                               weight_decay=settings["weight_decay"], dampening=0.0, nesterov=False)
    return torch.optim.Adam(params, lr=lr, betas=settings["betas"], eps=1e-8,
                            weight_decay=settings["weight_decay"], amsgrad=False)


class EarlyStopping:
    """Stop when validation accuracy has not improved for ``patience`` epochs, never before ``min_epochs``."""

    def __init__(self, patience=EARLY_STOP_PATIENCE, min_epochs=MIN_EPOCHS):
        self.patience = patience
        self.min_epochs = min_epochs
        self.best = None
        self.bad_epochs = 0

    def step(self, val_acc, epoch):
        """Record one epoch; return True when training should stop."""
        if self.best is None or val_acc > self.best:
            self.best = val_acc
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return epoch >= self.min_epochs and self.bad_epochs >= self.patience


def train_one_epoch(model, loader, optimizer, criterion):
    """One pass over the training split. Returns ``(mean loss, accuracy)``."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("the training loss is not a finite number")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item()) * int(labels.shape[0])
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        seen += int(labels.shape[0])
    return total_loss / seen, correct / seen


def validate(model, loader, criterion):
    """Loss and accuracy on the validation split, without gradients."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * int(labels.shape[0])
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += int(labels.shape[0])
    return total_loss / seen, correct / seen


def train(model, model_name, optimizer_name, run_name, loaders, backbone_lr_scale=None):
    """Train one model and save its history and its best checkpoint.

    Reseeds first: the initial weights were seeded when the model was built, and
    this fixes dropout and augmentation, so the result does not depend on which
    cells ran before.

    Returns:
        The training history, also saved as ``results/<run_name>/history.json``.
    """
    seed_everything(SEED)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, optimizer_name, backbone_lr_scale)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    stopper = EarlyStopping()
    checkpoint = CHECKPOINT_DIR / run_name / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    epochs, best_val_acc, best_epoch, total_time = [], None, None, 0.0
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion)
        val_loss, val_acc = validate(model, loaders["val"], criterion)
        scheduler.step()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)  # GPU work is asynchronous: wait before reading the clock
        epoch_time = time.perf_counter() - started
        total_time += epoch_time
        epochs.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc,
            "lr": max(float(group["lr"]) for group in optimizer.param_groups),
            "epoch_time_s": epoch_time,
        })

        improved = best_val_acc is None or val_acc > best_val_acc
        if improved:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save({"model": model_name, "epoch": epoch, "val_acc": val_acc,
                        "state_dict": model.state_dict()}, checkpoint)
        print(f"epoch {epoch:2d}/{EPOCHS}  train loss {train_loss:.4f}  acc {train_acc:.4f}  |  "
              f"val loss {val_loss:.4f}  acc {val_acc:.4f}  |  {epoch_time:5.1f} s"
              + ("  best" if improved else ""))
        if stopper.step(val_acc, epoch):
            print(f"Early stop: no improvement for {stopper.patience} epochs.")
            break

    settings = {key: list(value) if isinstance(value, tuple) else value
                for key, value in OPTIMIZER_SETTINGS[optimizer_name].items()}
    history = {
        "run": run_name,
        "model": model_name,
        "optimizer": optimizer_name,
        "seed": SEED,
        "device": describe_device(DEVICE),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "total_train_time_s": total_time,
        "hyperparameters": {
            **settings,
            "backbone_lr_scale": backbone_lr_scale,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "scheduler": f"cosine, min lr {MIN_LR}",
            "augmentation": "random horizontal flip, vertical flip, quarter turn (training only)",
        },
    }
    write_json(RESULTS_DIR / run_name / "history.json", history)
    print(f"Best validation accuracy {best_val_acc:.4f} at epoch {best_epoch}; "
          f"{total_time / len(epochs):.1f} s per epoch on {describe_device(DEVICE)}")
    return history
```
