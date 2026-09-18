"""Download EuroSAT, resize, draw the split, write the committed artifacts.

Owner: Member 1.   Stage: data   (SEAM 1 producer)

    IN   configs/stages/data.yaml
    OUT  data/processed/eurosat_64/        git-ignored
         data/splits/split_manifest.csv    COMMITTED
         data/splits/split_meta.json       COMMITTED
         data/splits/norm_stats.json       COMMITTED

Run once, by one person. The outputs are committed, so the other three members
never run this - they just pull.

    python scripts/prepare_data.py --config configs/stages/data.yaml
    python scripts/prepare_data.py --config configs/stages/data.yaml --synthetic
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stages/data.yaml")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate the tiny fixture dataset instead of downloading EuroSAT. "
             "This is the Phase 0 unblocker for Members 3 and 4.",
    )
    parser.add_argument("--force", action="store_true", help="Redraw an existing split.")
    args = parser.parse_args()

    # Member 1:
    #   1. load_config(args.config)
    #   2. seed_everything(cfg.seed)
    #   3. prepare_dataset(cfg)  (or make_synthetic_fixture when --synthetic)
    #   4. validate all three outputs against their schemas before exiting
    #
    # --force must be required to overwrite an existing manifest. Silently
    # redrawing the split would invalidate every committed result in
    # results/metrics/ without anyone noticing until the numbers stopped
    # reproducing.
    raise NotImplementedError("Member 1: implement scripts/prepare_data.py")


if __name__ == "__main__":
    raise SystemExit(main())
