from __future__ import annotations

import json
from pathlib import Path
import tempfile

from football_ai.detection.ball_detector_yolo import export_yolo_labels_from_coco_dataset


def test_exports_rectangular_coco_boxes_and_empty_negatives_to_yolo() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dataset = Path(directory)
        (dataset / "dataset_summary.json").write_text(
            json.dumps({"validation_sources": ["b"], "splits": {}}), encoding="utf-8"
        )
        for split in ("train", "valid"):
            split_dir = dataset / split
            (split_dir / "images").mkdir(parents=True)
            payload = {
                "images": [
                    {"id": 1, "file_name": "images/positive.jpg", "width": 100, "height": 80},
                    {"id": 2, "file_name": "images/negative.jpg", "width": 100, "height": 80},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [20, 10, 30, 10]}
                ],
                "categories": [{"id": 1, "name": "sports ball"}],
            }
            (split_dir / "_annotations.coco.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

        summary = export_yolo_labels_from_coco_dataset(dataset)

        assert summary["splits"]["train"]["positive_images"] == 1
        assert (dataset / "train/labels/positive.txt").read_text() == (
            "0 0.35000000 0.18750000 0.30000000 0.12500000\n"
        )
        assert (dataset / "train/labels/negative.txt").read_text() == ""
