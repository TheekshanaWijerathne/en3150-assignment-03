"""Evaluate a trained checkpoint on the held-out test split.

Owner: Member 1.   Stage: evaluation   (SEAM 4 producer)   Section 4 [25 marks]

    IN   configs/experiments/<name>.yaml
         artifacts/checkpoints/<run_id>/best.pt     (from Member 3)
         results/metrics/<run_id>/history.json      (for mean_epoch_time_s)
    OUT  results/metrics/<run_id>/test_metrics.json   COMMITTED
         results/metrics/<run_id>/resources.json      COMMITTED

    python scripts/evaluate.py --config configs/experiments/model_b__adam.yaml
    python scripts/evaluate.py --all          # every run with a checkpoint

This script is run for EVERY model - custom and pretrained alike - so all four
rows of the Section 6 table come from one code path. That is the whole point:
four members measuring their own model four ways produces a table that cannot
be defended.

This is the first and only time the test split is read.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Experiment config for one run.")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate every run that has a best.pt.")
    parser.add_argument("--latency-device", default="cpu",
                        help="Keep this cpu for every model - the Section 6 argument "
                             "is about edge deployment, and edge devices have no GPU.")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Metrics only; skip the resource profiling pass.")
    args = parser.parse_args()

    # Member 1:
    #   1. cfg = load_config(...); data = build_dataloaders(cfg)
    #   2. schema.validate_checkpoint(paths.best_checkpoint(cfg.run_id))
    #   3. model = build_model(...); load_state_dict
    #   4. y_true, y_pred = predict(model, data.test, device)
    #   5. compute_metrics(...)  -> test_metrics.json   via schema.write_json
    #   6. profile_model(...)    -> resources.json      via schema.write_json
    #
    # Assert that the confusion matrix sums to split_meta.json counts.test.
    # If it does not, the wrong split was evaluated - far better caught here
    # than in the viva.
    raise NotImplementedError("Member 1: implement scripts/evaluate.py")


if __name__ == "__main__":
    raise SystemExit(main())
