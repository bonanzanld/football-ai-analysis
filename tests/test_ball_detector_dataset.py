from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from football_ai.detection.ball_detector_dataset import (
    COCO_SPORTS_BALL_SLOT,
    detector_category_schema,
    export_coco_ball_detector_dataset,
    export_tiled_coco_ball_detector_dataset,
    extract_square_tile,
    load_human_detector_frames,
    square_tile_origins,
)


class BallDetectorDatasetTests(unittest.TestCase):
    def test_preserved_coco_schema_keeps_sports_ball_at_logit_slot_37(self) -> None:
        categories, ball_category_id = detector_category_schema(True)

        self.assertEqual(len(categories), 90)
        self.assertEqual(ball_category_id, COCO_SPORTS_BALL_SLOT)
        self.assertEqual(categories[COCO_SPORTS_BALL_SLOT]["name"], "sports ball")
        self.assertEqual([item["id"] for item in categories], list(range(90)))

    def _manifest(
        self,
        root: Path,
        name: str,
        source_video: str,
        annotations: list[dict[str, object]],
    ) -> Path:
        manifest_dir = root / name
        frames_dir = manifest_dir / "frames"
        frames_dir.mkdir(parents=True)
        for item in annotations:
            image = frames_dir / f"frame_{int(item['frame_number']):06d}.jpg"
            cv2.imwrite(str(image), np.zeros((80, 120, 3), dtype=np.uint8))
            item["image"] = f"frames/{image.name}"
        path = manifest_dir / "annotations.json"
        path.write_text(
            json.dumps({"source_video": source_video, "annotations": annotations}),
            encoding="utf-8",
        )
        return path

    def test_loads_only_visible_and_not_visible_human_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._manifest(
                root,
                "clip_a",
                "videos/a.mp4",
                [
                    {"frame_number": 1, "visibility": "visible", "ball_box": [10, 20, 20, 30], "review_status": "human_reviewed"},
                    {"frame_number": 2, "visibility": "occluded", "ball_box": [11, 21, 21, 31], "review_status": "human_reviewed"},
                    {"frame_number": 3, "visibility": "not_visible", "ball_box": None, "review_status": "human_reviewed"},
                    {"frame_number": 4, "visibility": "visible", "ball_box": [12, 22, 22, 32], "review_status": "unreviewed"},
                ],
            )

            frames = load_human_detector_frames([path])

            self.assertEqual([item.frame_number for item in frames], [1, 3])

    def test_exports_clip_separated_coco_with_negative_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self._manifest(
                root,
                "clip_a",
                "videos/a.mp4",
                [
                    {"frame_number": 1, "visibility": "visible", "ball_box": [10, 20, 20, 30], "review_status": "human_reviewed"},
                    {"frame_number": 2, "visibility": "not_visible", "ball_box": None, "review_status": "human_reviewed"},
                ],
            )
            validation = self._manifest(
                root,
                "clip_b",
                "videos/b.mp4",
                [
                    {"frame_number": 5, "visibility": "visible", "ball_box": [30, 40, 50, 60], "review_status": "human_reviewed"},
                ],
            )
            frames = load_human_detector_frames([train, validation])

            summary = export_coco_ball_detector_dataset(
                frames,
                root / "export",
                validation_sources=["videos/b.mp4"],
            )

            self.assertEqual(summary["splits"]["train"]["images"], 2)
            self.assertEqual(summary["splits"]["train"]["negative_images"], 1)
            self.assertEqual(summary["splits"]["valid"]["positive_images"], 1)
            payload = json.loads(
                (root / "export/valid/_annotations.coco.json").read_text()
            )
            self.assertEqual(payload["annotations"][0]["bbox"], [30.0, 40.0, 20.0, 20.0])
            self.assertEqual(
                {item["source_video"] for item in payload["images"]},
                {"videos/b.mp4"},
            )

    def test_coco_head_export_uses_ball_slot_37_without_reinitializing_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self._manifest(
                root,
                "clip_a",
                "videos/a.mp4",
                [{"frame_number": 1, "visibility": "visible", "ball_box": [10, 20, 20, 30], "review_status": "human_reviewed"}],
            )
            validation = self._manifest(
                root,
                "clip_b",
                "videos/b.mp4",
                [{"frame_number": 2, "visibility": "not_visible", "ball_box": None, "review_status": "human_reviewed"}],
            )

            summary = export_coco_ball_detector_dataset(
                load_human_detector_frames([train, validation]),
                root / "export",
                validation_sources=["videos/b.mp4"],
                preserve_coco_head=True,
            )
            payload = json.loads((root / "export/train/_annotations.coco.json").read_text())

            self.assertEqual(summary["class_layout"], "coco_90_preserved")
            self.assertEqual(len(payload["categories"]), 90)
            self.assertEqual(payload["annotations"][0]["category_id"], 37)

    def test_rejects_unknown_validation_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._manifest(
                root,
                "clip_a",
                "videos/a.mp4",
                [{"frame_number": 1, "visibility": "not_visible", "ball_box": None, "review_status": "human_reviewed"}],
            )
            frames = load_human_detector_frames([path])

            with self.assertRaisesRegex(ValueError, "Unknown validation source"):
                export_coco_ball_detector_dataset(
                    frames,
                    root / "export",
                    validation_sources=["videos/missing.mp4"],
                )

    def test_square_tiles_cover_edges_and_pad_small_images(self) -> None:
        origins = square_tile_origins(120, 80, tile_size=64, overlap=0.5)

        self.assertEqual(origins, ((0, 0), (32, 0), (56, 0), (0, 16), (32, 16), (56, 16)))
        image = np.ones((40, 50, 3), dtype=np.uint8)
        tile = extract_square_tile(image, (0, 0), tile_size=64)
        self.assertEqual(tile.shape, (64, 64, 3))
        self.assertTrue(np.all(tile[:40, :50] == 1))
        self.assertTrue(np.all(tile[40:, :] == 0))

    def test_tiled_export_keeps_sources_separate_and_skips_cut_ball(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self._manifest(
                root,
                "clip_a",
                "videos/a.mp4",
                [{"frame_number": 1, "visibility": "visible", "ball_box": [55, 30, 65, 40], "review_status": "human_reviewed"}],
            )
            validation = self._manifest(
                root,
                "clip_b",
                "videos/b.mp4",
                [{"frame_number": 2, "visibility": "not_visible", "ball_box": None, "review_status": "human_reviewed"}],
            )
            frames = load_human_detector_frames([train, validation])

            summary = export_tiled_coco_ball_detector_dataset(
                frames,
                root / "tiles",
                validation_sources=["videos/b.mp4"],
                tile_size=64,
                overlap=0.5,
            )

            self.assertGreater(summary["splits"]["train"]["positive_images"], 0)
            self.assertGreater(summary["splits"]["train"]["skipped_partial_ball_tiles"], 0)
            payload = json.loads((root / "tiles/train/_annotations.coco.json").read_text())
            self.assertEqual(
                {item["source_video"] for item in payload["images"]}, {"videos/a.mp4"}
            )
            self.assertTrue(all(item["bbox"][2] == 10 for item in payload["annotations"]))


if __name__ == "__main__":
    unittest.main()
