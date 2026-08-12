import json
from pathlib import Path
import tempfile
import unittest

from football_ai.detection.active_ball_dataset import (
    active_ball_dataset_report,
    aggregate_active_ball_dataset_reports,
    label_active_ball_candidates,
)
from football_ai.detection.ball_evaluation import BallGroundTruth
from tools.build_active_ball_dataset import _merge_annotations


class ActiveBallDatasetTests(unittest.TestCase):
    @staticmethod
    def _write_manifest(
        path: Path,
        visibility: str,
        review_status: str = "human_reviewed",
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_video": "videos/clip.mp4",
                    "annotations": [
                        {
                            "frame_number": 7,
                            "image": "frames/frame_000007.jpg",
                            "visibility": visibility,
                            "ball_box": (
                                None
                                if visibility in {"not_visible", "unreviewed"}
                                else [10, 10, 20, 20]
                            ),
                            "occlusion": "none",
                            "review_status": review_status,
                            "notes": "",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_merge_requires_explicit_permission_for_conflicting_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            self._write_manifest(first, "visible")
            self._write_manifest(second, "not_visible")

            with self.assertRaisesRegex(ValueError, "conflicting-overrides"):
                _merge_annotations([first, second])

    def test_merge_records_later_annotation_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            self._write_manifest(first, "visible")
            self._write_manifest(second, "not_visible")

            source, annotations, overridden = _merge_annotations(
                [first, second], allow_conflicting_overrides=True
            )

        self.assertEqual(source, "videos/clip.mp4")
        self.assertEqual(annotations[0].visibility, "not_visible")
        self.assertEqual(overridden, [7])

    def test_merge_keeps_reviewed_annotation_over_later_open_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reviewed = Path(directory) / "reviewed.json"
            open_dense = Path(directory) / "open-dense.json"
            self._write_manifest(reviewed, "visible")
            self._write_manifest(open_dense, "unreviewed", "unreviewed")

            _, annotations, overridden = _merge_annotations([reviewed, open_dense])

        self.assertEqual(annotations[0].visibility, "visible")
        self.assertEqual(annotations[0].review_status, "human_reviewed")
        self.assertEqual(overridden, [])

    def test_merge_promotes_later_review_over_open_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            open_dense = Path(directory) / "open-dense.json"
            reviewed = Path(directory) / "reviewed.json"
            self._write_manifest(open_dense, "unreviewed", "unreviewed")
            self._write_manifest(reviewed, "visible")

            _, annotations, overridden = _merge_annotations([open_dense, reviewed])

        self.assertEqual(annotations[0].visibility, "visible")
        self.assertEqual(annotations[0].review_status, "human_reviewed")
        self.assertEqual(overridden, [])

    def test_labels_matches_far_negatives_and_near_misses_definitively(self) -> None:
        annotations = [
            BallGroundTruth(1, "visible", (10, 10, 20, 20)),
            BallGroundTruth(2, "not_visible"),
            BallGroundTruth(3, "occluded", (30, 30, 40, 40), "player"),
        ]
        examples = label_active_ball_candidates(
            annotations,
            {
                1: [((11, 11, 21, 21), 0.8), ((80, 80, 90, 90), 0.2)],
                2: [((50, 50, 60, 60), 0.7)],
                3: [((100, 100, 110, 110), 0.6)],
            },
        )

        self.assertEqual(
            [item.label for item in examples],
            ["positive", "negative", "negative", "negative"],
        )

    def test_keeps_near_nonmatching_candidate_ambiguous(self) -> None:
        examples = label_active_ball_candidates(
            [BallGroundTruth(1, "visible", (10, 10, 20, 20))],
            {1: [((35, 10, 45, 20), 0.4)]},
        )

        self.assertEqual(examples[0].label, "ambiguous")

    def test_report_refuses_small_single_clip_dataset(self) -> None:
        annotations = [BallGroundTruth(1, "visible", (10, 10, 20, 20))]
        examples = label_active_ball_candidates(
            annotations,
            {1: [((10, 10, 20, 20), 0.8)]},
        )

        report = active_ball_dataset_report(
            annotations,
            examples,
            source_clip_count=1,
        )

        self.assertFalse(report["ready_for_training"])
        self.assertEqual(report["positive_frames"], 1)
        self.assertEqual(report["missing_positive_frames"], [])
        self.assertEqual(len(report["blocking_reasons"]), 2)

    def test_report_blocks_visible_ball_frame_without_candidate(self) -> None:
        annotations = [BallGroundTruth(7, "visible", (10, 10, 20, 20))]

        report = active_ball_dataset_report(
            annotations,
            [],
            source_clip_count=3,
            minimum_positive_frames=0,
        )

        self.assertEqual(report["missing_positive_frames"], [7])
        self.assertFalse(report["ready_for_training"])

    def test_report_diagnoses_but_does_not_block_occluded_frame_without_candidate(
        self,
    ) -> None:
        annotations = [BallGroundTruth(7, "occluded", (10, 10, 20, 20), "player")]

        report = active_ball_dataset_report(
            annotations,
            [],
            source_clip_count=3,
            minimum_positive_frames=0,
        )

        self.assertEqual(report["missing_positive_frames"], [])
        self.assertEqual(report["occluded_without_positive_frames"], [7])
        self.assertTrue(report["ready_for_training"])

    def test_aggregate_requires_distinct_sources_and_combines_evidence(self) -> None:
        reports = [
            {
                "source_video": f"videos/clip-{index}.mp4",
                "reviewed_frames": 50,
                "candidate_examples": 80,
                "label_counts": {"positive": 40, "negative": 40},
                "positive_frames": 40,
                "missing_positive_frames": [],
                "occluded_without_positive_frames": [],
            }
            for index in range(3)
        ]

        aggregate = aggregate_active_ball_dataset_reports(reports)

        self.assertTrue(aggregate["ready_for_training"])
        self.assertEqual(aggregate["source_clips"], 3)
        self.assertEqual(aggregate["positive_source_clips"], 3)
        self.assertEqual(aggregate["positive_frames"], 120)
        self.assertEqual(aggregate["label_counts"], {"negative": 120, "positive": 120})

    def test_aggregate_does_not_count_empty_reports_as_positive_sources(self) -> None:
        reports = [
            {
                "source_video": f"videos/clip-{index}.mp4",
                "reviewed_frames": 40 if index < 2 else 0,
                "candidate_examples": 40 if index < 2 else 0,
                "label_counts": {"positive": 40} if index < 2 else {},
                "positive_frames": 60 if index < 2 else 0,
                "missing_positive_frames": [],
                "occluded_without_positive_frames": [],
            }
            for index in range(3)
        ]

        aggregate = aggregate_active_ball_dataset_reports(reports)

        self.assertEqual(aggregate["source_clips"], 3)
        self.assertEqual(aggregate["positive_source_clips"], 2)
        self.assertFalse(aggregate["ready_for_training"])
        self.assertIn(
            "only 2 source clips with positive frames; require at least 3",
            aggregate["blocking_reasons"],
        )

    def test_aggregate_rejects_duplicate_source_video(self) -> None:
        report = {
            "source_video": "videos/clip.mp4",
            "label_counts": {},
            "missing_positive_frames": [],
            "occluded_without_positive_frames": [],
        }

        with self.assertRaisesRegex(ValueError, "Duplicate source"):
            aggregate_active_ball_dataset_reports([report, report])


if __name__ == "__main__":
    unittest.main()
