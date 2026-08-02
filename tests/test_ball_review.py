from pathlib import Path
import tempfile
import unittest

from football_ai.detection.ball_review import (
    centered_image_box,
    confirm_ball_annotation,
    image_box_from_drag,
    review_progress,
    save_review_manifest,
)


class BallReviewTests(unittest.TestCase):
    def test_single_click_creates_centered_original_image_box(self) -> None:
        box = centered_image_box(
            (110, 70),
            size=20.0,
            scale=0.5,
            offset_x=100,
            offset_y=50,
            image_width=1920,
            image_height=1080,
        )
        self.assertEqual(box, (10.0, 30.0, 30.0, 50.0))

    def test_converts_display_drag_to_original_image_box(self) -> None:
        box = image_box_from_drag(
            (110, 70),
            (130, 90),
            scale=0.5,
            offset_x=100,
            offset_y=50,
            image_width=1920,
            image_height=1080,
        )
        self.assertEqual(box, (20.0, 40.0, 60.0, 80.0))

    def test_rejects_tiny_drag(self) -> None:
        box = image_box_from_drag(
            (10, 10),
            (10, 10),
            scale=1.0,
            offset_x=0,
            offset_y=0,
            image_width=100,
            image_height=100,
        )
        self.assertIsNone(box)

    def test_confirm_marks_annotation_as_human_reviewed(self) -> None:
        result = confirm_ball_annotation(
            {"frame_number": 7, "visibility": "unreviewed"},
            visibility="occluded",
            box=(1, 2, 3, 4),
            occlusion="player",
        )
        self.assertEqual(result["review_status"], "human_reviewed")
        self.assertEqual(result["ball_box"], [1, 2, 3, 4])

    def test_not_visible_clears_box_and_occlusion(self) -> None:
        result = confirm_ball_annotation(
            {"frame_number": 7},
            visibility="not_visible",
            box=(1, 2, 3, 4),
            occlusion="shadow",
        )
        self.assertIsNone(result["ball_box"])
        self.assertEqual(result["occlusion"], "none")

    def test_save_sets_completion_only_when_every_frame_reviewed(self) -> None:
        payload = {
            "annotations": [
                {"review_status": "human_reviewed"},
                {"review_status": "unreviewed"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            save_review_manifest(payload, path)
        self.assertFalse(payload["human_review_complete"])
        self.assertEqual(review_progress(payload["annotations"]), (1, 2))


if __name__ == "__main__":
    unittest.main()
