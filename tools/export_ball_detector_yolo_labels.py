from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detection.ball_detector_yolo import export_yolo_labels_from_coco_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add YOLO labels to an existing COCO ball-detector dataset."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()
    summary = export_yolo_labels_from_coco_dataset(args.dataset)
    for split, metrics in summary["splits"].items():
        print(
            f"{split}: images={metrics['images']} | "
            f"positive={metrics['positive_images']} | "
            f"negative={metrics['negative_images']}"
        )


if __name__ == "__main__":
    main()
