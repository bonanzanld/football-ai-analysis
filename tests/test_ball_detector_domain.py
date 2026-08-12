from __future__ import annotations

import json

import cv2
import numpy as np

from football_ai.detection.ball_detector_domain import (
    analyze_coco_ball_domain,
    measure_ball_box,
)


def test_measure_ball_box_detects_bright_foreground() -> None:
    image = np.full((30, 30, 3), 20, dtype=np.uint8)
    image[10:20, 10:20] = 220

    metrics = measure_ball_box(image, [10, 10, 10, 10])

    assert metrics["width"] == 10
    assert metrics["aspect_ratio"] == 1
    assert metrics["brightness"] == 220
    assert metrics["foreground_contrast"] == 200


def test_analyze_coco_ball_domain_groups_split_and_source(tmp_path) -> None:
    for split, brightness, source in (
        ("train", 200, "bright.mp4"),
        ("valid", 80, "dark.mp4"),
    ):
        image_dir = tmp_path / split / "images"
        image_dir.mkdir(parents=True)
        image = np.full((24, 24, 3), 20, dtype=np.uint8)
        image[8:16, 8:16] = brightness
        cv2.imwrite(str(image_dir / "frame.jpg"), image)
        payload = {
            "images": [
                {
                    "id": 1,
                    "file_name": "images/frame.jpg",
                    "source_video": source,
                }
            ],
            "annotations": [{"image_id": 1, "bbox": [8, 8, 8, 8]}],
        }
        (tmp_path / split / "_annotations.coco.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    report = analyze_coco_ball_domain(tmp_path)

    assert report["splits"]["train"]["count"] == 1
    assert report["splits"]["valid"]["metrics"]["brightness"]["median"] == 80
    assert report["sources"]["dark.mp4"]["count"] == 1
