from pathlib import Path
import tempfile
import unittest

import numpy as np

from football_ai.detection.ball_review import (
    centered_image_box,
    confirm_ball_annotation,
    image_box_from_drag,
    next_required_review_index,
    next_unreviewed_index,
    propagate_ball_box_optical_flow,
    proposed_candidate_box,
    review_progress,
    save_review_manifest,
    visible_human_review_indices,
)


class BallReviewTests(unittest.TestCase):
    def test_visible_recheck_queue_only_contains_human_visible_boxes(self) -> None:
        annotations = [
            {"review_status": "human_reviewed", "visibility": "visible", "ball_box": [1, 2, 3, 4]},
            {"review_status": "human_reviewed", "visibility": "occluded", "ball_box": [1, 2, 3, 4]},
            {"review_status": "ai_draft", "visibility": "visible", "ball_box": [1, 2, 3, 4]},
            {"review_status": "human_reviewed", "visibility": "visible", "ball_box": None},
            {"review_status": "human_reviewed", "visibility": "visible", "ball_box": [5, 6, 7, 8]},
        ]

        self.assertEqual(visible_human_review_indices(annotations), [0, 4])

    def test_optical_flow_propagates_seed_without_human_review_claim(self) -> None:
        first = np.zeros((80, 120, 3), dtype=np.uint8)
        second = np.zeros_like(first)
        third = np.zeros_like(first)
        cv = __import__("cv2")
        cv.circle(first, (30, 40), 4, (255, 255, 255), -1)
        cv.circle(second, (34, 42), 4, (255, 255, 255), -1)
        cv.circle(third, (38, 44), 4, (255, 255, 255), -1)

        boxes = propagate_ball_box_optical_flow(
            [first, second, third], seed_index=0, seed_box=(26, 36, 34, 44)
        )

        self.assertIsNotNone(boxes[2])
        assert boxes[2] is not None
        self.assertAlmostEqual((boxes[2][0] + boxes[2][2]) / 2, 38, delta=0.5)
        self.assertAlmostEqual((boxes[2][1] + boxes[2][3]) / 2, 44, delta=0.5)

    def test_required_review_skips_optional_ai_drafts(self) -> None:
        annotations = [
            {"review_status": "human_reviewed"},
            {"review_status": "ai_draft", "review_priority": "optional"},
            {"review_status": "ai_draft", "review_priority": "required"},
        ]

        self.assertEqual(next_required_review_index(annotations), 2)

    def test_required_review_still_includes_unreviewed_frames(self) -> None:
        annotations = [
            {"review_status": "ai_draft", "review_priority": "optional"},
            {"review_status": "unreviewed"},
        ]

        self.assertEqual(next_required_review_index(annotations), 1)
    def test_next_unreviewed_resumes_and_wraps(self) -> None:
        annotations = [
            {"review_status": "human_reviewed"},
            {"review_status": "unreviewed"},
            {"review_status": "human_reviewed"},
            {"review_status": "unreviewed"},
        ]

        self.assertEqual(next_unreviewed_index(annotations), 1)
        self.assertEqual(next_unreviewed_index(annotations, after=1), 3)
        self.assertEqual(next_unreviewed_index(annotations, after=3), 1)

    def test_next_unreviewed_returns_none_when_complete(self) -> None:
        self.assertIsNone(
            next_unreviewed_index([{"review_status": "human_reviewed"}])
        )

    def test_proposes_highest_confidence_candidate_in_image_space(self) -> None:
        box = proposed_candidate_box(
            {
                "candidates": [
                    {"box": [1, 2, 11, 12], "confidence": 0.4},
                    {"box": [20, 30, 40, 50], "confidence": 0.8},
                ],
                "transform": np.eye(3).tolist(),
            },
            image_width=100,
            image_height=100,
        )

        self.assertEqual(box, (20.0, 30.0, 40.0, 50.0))

    def test_proposal_returns_none_without_candidates(self) -> None:
        self.assertIsNone(
            proposed_candidate_box(
                {"candidates": [], "transform": np.eye(3).tolist()},
                image_width=100,
                image_height=100,
            )
        )

    def test_proposal_rejects_weak_highest_confidence_candidate(self) -> None:
        self.assertIsNone(
            proposed_candidate_box(
                {
                    "candidates": [
                        {"box": [20, 30, 40, 50], "confidence": 0.36},
                    ],
                    "transform": np.eye(3).tolist(),
                },
                image_width=100,
                image_height=100,
            )
        )

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
