from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detection.ball_detector_dataset import (
    export_coco_ball_detector_dataset,
    export_tiled_coco_ball_detector_dataset,
    load_human_detector_frames,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export human-reviewed ball boxes as clip-separated COCO data."
    )
    parser.add_argument("--annotations", nargs="+", required=True, type=Path)
    parser.add_argument("--validation-source", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument(
        "--preserve-coco-head",
        action="store_true",
        help="Keep RF-DETR's 90 pretrained logit slots and sports-ball slot 37.",
    )
    args = parser.parse_args()

    frames = load_human_detector_frames(args.annotations)
    if args.tile_size is None:
        summary = export_coco_ball_detector_dataset(
            frames,
            args.output,
            validation_sources=args.validation_source,
            preserve_coco_head=args.preserve_coco_head,
        )
    else:
        summary = export_tiled_coco_ball_detector_dataset(
            frames,
            args.output,
            validation_sources=args.validation_source,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
            preserve_coco_head=args.preserve_coco_head,
        )
    for split, metrics in summary["splits"].items():
        print(
            f"{split}: images={metrics['images']} | "
            f"positive={metrics['positive_images']} | "
            f"negative={metrics['negative_images']}"
        )
    print(f"Dataset: {args.output}")


if __name__ == "__main__":
    main()
