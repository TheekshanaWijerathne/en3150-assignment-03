"""Train one model under one optimizer.

Owner: Member 3.   Stage: training   (SEAM 3 producer)   Section 4 [25 marks]

    IN   configs/experiments/<name>.yaml
         SEAM 1: build_dataloaders(cfg) -> DataBundle
         SEAM 2: build_model(cfg.model_name, ...)
    OUT  artifacts/checkpoints/<run_id>/best.pt   git-ignored
         results/metrics/<run_id>/history.json    COMMITTED
         results/figures/<run_id>/curves.png      COMMITTED

    python scripts/train.py --config configs/experiments/model_b__adam.yaml
    python scripts/train.py --config configs/experiments/model_b__adam.yaml \
        --epochs 2 --subset-fraction 0.05          # fast smoke test
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:N")
    parser.add_argument("--subset-fraction", type=float, default=None,
                        help="Train on a slice for fast iteration. Results from a run "
                             "with this below 1.0 must NOT be committed.")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt.")
    args = parser.parse_args()

    # Member 3:
    #   1. cfg = load_config(args.config, **overrides)
    #   2. setup_logging(); seed_everything(cfg.seed)
    #   3. data  = build_dataloaders(cfg)                     SEAM 1
    #   4. model = build_model(cfg.model_name, data.num_classes, data.input_shape)
    #   5. result = Trainer(cfg).fit(model, data, cfg)        SEAM 3
    #   6. plot_loss_curves(cfg.run_id)
    #
    # Refuse to write into results/metrics/ when subset_fraction < 1.0 or when
    # epochs < 20. Both would produce a schema-invalid or misleading artifact,
    # and it is much cheaper to refuse here than to discover it during report
    # assembly. Print the run_id and every output path on exit so the next
    # member knows exactly what to read.
    raise NotImplementedError("Member 3: implement scripts/train.py")


if __name__ == "__main__":
    raise SystemExit(main())
