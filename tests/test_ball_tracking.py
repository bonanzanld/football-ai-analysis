from pathlib import Path
import json
import tempfile
import unittest

from football_ai.detection.ball_tracking import (
    BallCandidate,
    BallObservation,
    BallTracker,
    exclude_candidates_inside_people,
    interpolate_detected_gaps,
    save_ball_observations,
)


class BallTrackerTests(unittest.TestCase):
    def test_prefers_temporally_near_candidate_over_distant_false_positive(self) -> None:
        tracker = BallTracker(
            maximum_jump_pixels=80.0,
            acquisition_confidence=0.30,
        )
        tracker.update(0, [BallCandidate((10, 10, 20, 20), 0.80)])
        selected = tracker.update(
            1,
            [
                BallCandidate((18, 10, 28, 20), 0.65),
                BallCandidate((300, 300, 312, 312), 0.99),
            ],
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "detected")
        self.assertAlmostEqual(selected.center[0], 23.0)

    def test_bridges_short_gap_and_labels_prediction_honestly(self) -> None:
        tracker = BallTracker(maximum_gap_frames=2)
        tracker.update(0, [BallCandidate((10, 10, 20, 20), 0.90)])
        tracker.update(1, [BallCandidate((20, 10, 30, 20), 0.80)])
        predicted = tracker.update(2, [])
        self.assertIsNotNone(predicted)
        self.assertEqual(predicted.source, "predicted")
        self.assertAlmostEqual(predicted.center[0], 35.0)

    def test_stops_predicting_after_configured_gap(self) -> None:
        tracker = BallTracker(maximum_gap_frames=1)
        tracker.update(0, [BallCandidate((10, 10, 20, 20), 0.90)])
        self.assertIsNotNone(tracker.update(1, []))
        self.assertIsNone(tracker.update(2, []))

    def test_hides_prediction_below_fifteen_percent(self) -> None:
        tracker = BallTracker(maximum_gap_frames=10)
        tracker.update(0, [BallCandidate((10, 10, 20, 20), 0.50)])
        results = [tracker.update(frame, []) for frame in range(1, 6)]
        self.assertIsNotNone(results[0])
        self.assertIsNotNone(results[2])
        self.assertIsNone(results[3])
        self.assertIsNone(results[4])

    def test_rejects_implausibly_large_candidate(self) -> None:
        tracker = BallTracker()
        result = tracker.update(0, [BallCandidate((0, 0, 500, 500), 0.99)])
        self.assertIsNone(result)

    def test_rejects_non_finite_candidate(self) -> None:
        tracker = BallTracker()
        result = tracker.update(0, [BallCandidate((10, 10, float("nan"), 20), 0.99)])
        self.assertIsNone(result)

    def test_reacquires_ball_after_long_gap(self) -> None:
        tracker = BallTracker(maximum_gap_frames=1, maximum_jump_pixels=40.0)
        tracker.update(0, [BallCandidate((10, 10, 20, 20), 0.90)])
        tracker.update(1, [])
        reacquired = tracker.update(2, [BallCandidate((300, 200, 310, 210), 0.70)])
        self.assertIsNotNone(reacquired)
        self.assertEqual(reacquired.source, "detected")

    def test_strong_detection_replaces_weak_wrong_track(self) -> None:
        tracker = BallTracker(
            maximum_jump_pixels=80.0,
            acquisition_confidence=0.30,
        )
        tracker.update(0, [BallCandidate((10, 10, 20, 20), 0.37)])
        pending = tracker.update(
            1,
            [
                BallCandidate((12, 10, 22, 20), 0.35),
                BallCandidate((900, 300, 910, 310), 0.68),
            ],
        )
        self.assertIsNone(pending)
        selected = tracker.update(
            2,
            [BallCandidate((910, 302, 920, 312), 0.70)],
        )
        self.assertIsNotNone(selected)
        self.assertAlmostEqual(selected.center[0], 915.0)

    def test_unconfirmed_distant_reacquisition_is_discarded(self) -> None:
        tracker = BallTracker(
            maximum_jump_pixels=80.0,
            acquisition_confidence=0.30,
        )
        tracker.update(0, [BallCandidate((10, 10, 20, 20), 0.40)])
        self.assertIsNone(
            tracker.update(1, [BallCandidate((900, 300, 910, 310), 0.70)])
        )
        result = tracker.update(2, [BallCandidate((300, 100, 310, 110), 0.72)])
        self.assertIsNone(result)

    def test_serializes_observations(self) -> None:
        tracker = BallTracker()
        tracker.update(0, [BallCandidate((10, 10, 20, 20), 0.90)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ball.json"
            save_ball_observations(tracker.observations, path, "match.mp4", 30.0)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["observations"][0]["source"], "detected")

    def test_weak_candidate_cannot_start_track(self) -> None:
        tracker = BallTracker(acquisition_confidence=0.50)
        self.assertIsNone(
            tracker.update(0, [BallCandidate((10, 10, 20, 20), 0.37)])
        )

    def test_strong_candidate_starts_track_after_weak_bootstrap_frames(self) -> None:
        tracker = BallTracker(acquisition_confidence=0.50)
        self.assertIsNone(
            tracker.update(0, [BallCandidate((10, 10, 20, 20), 0.37)])
        )
        selected = tracker.update(1, [BallCandidate((300, 200, 310, 210), 0.68)])
        self.assertIsNotNone(selected)
        self.assertAlmostEqual(selected.center[0], 305.0)

    def test_candidate_below_fifteen_percent_requires_nearby_track(self) -> None:
        tracker = BallTracker(maximum_jump_pixels=140.0)
        tracker.update(0, [BallCandidate((100, 100, 110, 110), 0.70)])
        selected = tracker.update(1, [BallCandidate((180, 100, 190, 110), 0.06)])
        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "predicted")
        self.assertAlmostEqual(selected.center[0], 105.0)

    def test_candidate_below_fifteen_percent_can_support_nearby_track(self) -> None:
        tracker = BallTracker(maximum_jump_pixels=140.0)
        tracker.update(0, [BallCandidate((100, 100, 110, 110), 0.70)])
        selected = tracker.update(1, [BallCandidate((112, 100, 122, 110), 0.06)])
        self.assertIsNotNone(selected)

    def test_expected_zone_grows_after_a_short_occlusion(self) -> None:
        tracker = BallTracker(
            maximum_jump_pixels=40.0,
            acquisition_confidence=0.30,
        )
        tracker.update(0, [BallCandidate((100, 100, 110, 110), 0.80)])
        tracker.update(1, [])
        selected = tracker.update(
            2,
            [BallCandidate((138, 100, 148, 110), 0.40)],
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "detected")

    def test_expected_zone_remains_bounded(self) -> None:
        tracker = BallTracker(
            maximum_gap_frames=5,
            maximum_jump_pixels=40.0,
            acquisition_confidence=0.30,
        )
        tracker.update(0, [BallCandidate((100, 100, 110, 110), 0.80)])
        tracker.update(1, [])
        selected = tracker.update(
            2,
            [BallCandidate((300, 100, 310, 110), 0.40)],
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "predicted")

    def test_prediction_does_not_feed_back_into_velocity(self) -> None:
        tracker = BallTracker(maximum_gap_frames=5)
        tracker.update(0, [BallCandidate((100, 100, 110, 110), 0.80)])
        tracker.update(1, [BallCandidate((110, 100, 120, 110), 0.80)])
        predictions = [tracker.update(frame, []) for frame in range(2, 7)]
        centers = [item.center[0] for item in predictions if item is not None]
        self.assertEqual(centers, [125.0, 135.0, 145.0, 155.0, 165.0])

    def test_rejects_candidate_inside_person_lower_body(self) -> None:
        candidate = BallCandidate((45, 75, 55, 85), 0.80)
        result = exclude_candidates_inside_people([candidate], [(20, 20, 80, 100)])
        self.assertEqual(result, [])

    def test_keeps_candidate_beside_person_box(self) -> None:
        candidate = BallCandidate((85, 75, 95, 85), 0.80)
        result = exclude_candidates_inside_people([candidate], [(20, 20, 80, 100)])
        self.assertEqual(result, [candidate])

    def test_keeps_candidate_at_persons_feet(self) -> None:
        candidate = BallCandidate((45, 92, 55, 102), 0.80)
        result = exclude_candidates_inside_people([candidate], [(20, 20, 80, 100)])
        self.assertEqual(result, [candidate])

    def test_interpolates_short_gap_between_real_detections(self) -> None:
        tracker = BallTracker(maximum_gap_frames=1)
        tracker.update(0, [BallCandidate((10, 10, 20, 20), 0.90)])
        tracker.update(1, [])
        tracker.update(2, [])
        tracker.update(3, [BallCandidate((40, 10, 50, 20), 0.80)])

        result = interpolate_detected_gaps(tracker.observations)
        by_frame = {item.frame_number: item for item in result}

        self.assertEqual(by_frame[1].source, "interpolated")
        self.assertEqual(by_frame[2].source, "interpolated")
        self.assertAlmostEqual(by_frame[1].center[0], 25.0)
        self.assertAlmostEqual(by_frame[2].center[0], 35.0)

    def test_does_not_interpolate_long_gap(self) -> None:
        observations = (
            BallObservation(0, (10.0, 10.0), (5.0, 5.0, 15.0, 15.0), 0.9, "detected"),
            BallObservation(20, (20.0, 10.0), (15.0, 5.0, 25.0, 15.0), 0.9, "detected"),
        )
        result = interpolate_detected_gaps(observations, maximum_gap_frames=12)
        self.assertEqual(len(result), 2)

    def test_does_not_interpolate_implausible_jump(self) -> None:
        observations = (
            BallObservation(0, (10.0, 10.0), (5.0, 5.0, 15.0, 15.0), 0.9, "detected"),
            BallObservation(3, (400.0, 10.0), (395.0, 5.0, 405.0, 15.0), 0.9, "detected"),
        )
        result = interpolate_detected_gaps(
            observations,
            maximum_speed_pixels_per_frame=75.0,
        )
        self.assertEqual(len(result), 2)

    def test_interpolates_airborne_ball_hidden_against_similar_background(self) -> None:
        observations = (
            BallObservation(
                0,
                (200.0, 500.0),
                (190.0, 490.0, 210.0, 510.0),
                0.82,
                "detected",
            ),
            BallObservation(
                31,
                (820.0, 260.0),
                (812.0, 252.0, 828.0, 268.0),
                0.74,
                "detected",
            ),
        )

        result = interpolate_detected_gaps(
            observations,
            maximum_gap_frames=45,
            maximum_speed_pixels_per_frame=100.0,
        )
        by_frame = {item.frame_number: item for item in result}

        self.assertEqual(len(result), 32)
        self.assertEqual(by_frame[15].source, "interpolated")
        self.assertLess(by_frame[15].center[1], observations[0].center[1])
        self.assertGreater(by_frame[15].center[0], observations[0].center[0])


if __name__ == "__main__":
    unittest.main()
