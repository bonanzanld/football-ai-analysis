from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


METRIC_NAMES = (
    "width",
    "height",
    "aspect_ratio",
    "sharpness",
    "foreground_contrast",
    "local_std",
    "brightness",
)


def _percentiles(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(tuple(values), dtype=np.float64)
    return {
        "p10": float(np.percentile(data, 10)),
        "median": float(np.median(data)),
        "p90": float(np.percentile(data, 90)),
    }


def measure_ball_box(
    image: np.ndarray, bbox: Iterable[float]
) -> dict[str, float]:
    """Measure appearance inside a ball box and against its immediate ring."""

    x, y, width, height = (float(value) for value in bbox)
    if width <= 0 or height <= 0:
        raise ValueError("Ball box width and height must be positive")
    image_height, image_width = image.shape[:2]
    x1 = max(0, int(np.floor(x)))
    y1 = max(0, int(np.floor(y)))
    x2 = min(image_width, int(np.ceil(x + width)))
    y2 = min(image_height, int(np.ceil(y + height)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Ball box falls outside the image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ball = gray[y1:y2, x1:x2]
    margin_x = max(2, int(round(width)))
    margin_y = max(2, int(round(height)))
    rx1, ry1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    rx2, ry2 = min(image_width, x2 + margin_x), min(image_height, y2 + margin_y)
    local = gray[ry1:ry2, rx1:rx2]
    ring_mask = np.ones(local.shape, dtype=bool)
    ring_mask[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
    ring = local[ring_mask]
    ring_mean = float(np.mean(ring)) if ring.size else float(np.mean(ball))

    return {
        "width": width,
        "height": height,
        "aspect_ratio": max(width, height) / min(width, height),
        "sharpness": float(cv2.Laplacian(ball, cv2.CV_64F).var()),
        "foreground_contrast": abs(float(np.mean(ball)) - ring_mean),
        "local_std": float(np.std(local)),
        "brightness": float(np.mean(ball)),
    }


def analyze_coco_ball_domain(dataset_dir: str | Path) -> dict[str, object]:
    """Summarize ball appearance per split and source in a COCO dataset."""

    root = Path(dataset_dir)
    rows: list[dict[str, object]] = []
    for split in ("train", "valid"):
        annotation_path = root / split / "_annotations.coco.json"
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        images = {int(item["id"]): item for item in payload["images"]}
        for annotation in payload["annotations"]:
            image_info = images[int(annotation["image_id"])]
            image_path = root / split / str(image_info["file_name"])
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            rows.append(
                {
                    "split": split,
                    "source_video": str(image_info.get("source_video", "unknown")),
                    **measure_ball_box(image, annotation["bbox"]),
                }
            )

    def summarize(items: list[dict[str, object]]) -> dict[str, object]:
        return {
            "count": len(items),
            "metrics": {
                name: _percentiles(float(item[name]) for item in items)
                for name in METRIC_NAMES
            },
        }

    by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split"])].append(row)
        by_source[str(row["source_video"])].append(row)
    return {
        "schema_version": 1,
        "dataset": str(root),
        "splits": {key: summarize(value) for key, value in sorted(by_split.items())},
        "sources": {key: summarize(value) for key, value in sorted(by_source.items())},
    }
