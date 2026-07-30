from pathlib import Path
import json
import tempfile
import unittest

from football_ai.detection.ball_tracking import (
    BallCandidate,
    BallObservation,
    BallTracker,
    exclude_candidates_inside_people,
    hold_stationary_detected_gaps,
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

    def test_locked_track_prefers_near_lower_confidence_candidate(self) -> None:
        tracker = BallTracker(
            maximum_jump_pixels=70.0,
            maximum_speed_pixels_per_frame=70.0,
            acquisition_confidence=0.30,
        )
        tracker.update(0, [BallCandidate((95, 95, 105, 105), 0.72)])

        selected = tracker.update(
            1,
            [
                BallCandidate((99, 95, 109, 105), 0.32),
                BallCandidate((135, 95, 145, 105), 0.78),
            ],
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "detected")
        self.assertAlmostEqual(selected.center[0], 104.0)

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
        tracker = BallTracker(
            maximum_gap_frames=1,
            maximum_jump_pixels=40.0,
            unrestricted_reacquisition_after_frames=1,
        )
        tracker.update(0, [BallCandidate((10, 10, 20, 20), 0.90)])
        tracker.update(1, [])
        self.assertIsNone(
            tracker.update(2, [BallCandidate((300, 200, 310, 210), 0.70)])
        )
        self.assertIsNone(
            tracker.update(3, [BallCandidate((302, 201, 312, 211), 0.72)])
        )
        reacquired = tracker.update(4, [BallCandidate((304, 202, 314, 212), 0.74)])
        self.assertIsNotNone(reacquired)
        self.assertEqual(reacquired.source, "detected")

    def test_reappearance_prefers_last_known_area_over_higher_confidence(self) -> None:
        tracker = BallTracker(
            maximum_gap_frames=1,
            maximum_jump_pixels=80.0,
            acquisition_confidence=0.30,
            strong_reacquisition_confidence=0.35,
        )
        tracker.update(0, [BallCandidate((95, 95, 105, 105), 0.72)])
        tracker.update(1, [])
        selected = tracker.update(
            2,
            [
                BallCandidate((99, 96, 109, 106), 0.38),
                BallCandidate((155, 95, 165, 105), 0.91),
            ],
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "detected")
        self.assertAlmostEqual(selected.center[0], 104.0)

    def test_remote_reappearance_needs_three_consistent_frames(self) -> None:
        tracker = BallTracker(
            maximum_gap_frames=0,
            unrestricted_reacquisition_after_frames=1,
            acquisition_confidence=0.30,
            reacquisition_confirmation_frames=3,
        )
        tracker.update(0, [BallCandidate((95, 95, 105, 105), 0.72)])
        self.assertIsNone(
            tracker.update(2, [BallCandidate((395, 195, 405, 205), 0.82)])
        )
        self.assertIsNone(
            tracker.update(3, [BallCandidate((399, 197, 409, 207), 0.84)])
        )
        selected = tracker.update(
            4,
            [BallCandidate((403, 199, 413, 209), 0.86)],
        )
        self.assertIsNotNone(selected)
        self.assertAlmostEqual(selected.center[0], 408.0)

    def test_distant_candidate_is_rejected_before_long_reacquisition_gap(self) -> None:
        tracker = BallTracker(maximum_gap_frames=1, maximum_jump_pixels=40.0)
        tracker.update(0, [BallCandidate((10, 10, 20, 20), 0.90)])
        tracker.update(1, [])
        self.assertIsNone(
            tracker.update(2, [BallCandidate((300, 200, 310, 210), 0.95)])
        )

    def test_velocity_prediction_cannot_allow_impossible_single_frame_jump(self) -> None:
        tracker = BallTracker(
            maximum_jump_pixels=140.0,
            maximum_speed_pixels_per_frame=100.0,
        )
        tracker.update(0, [BallCandidate((0, 0, 10, 10), 0.90)])
        tracker.update(1, [BallCandidate((70, 0, 80, 10), 0.90)])
        result = tracker.update(2, [BallCandidate((210, 0, 220, 10), 0.90)])
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "predicted")
        self.assertNotAlmostEqual(result.center[0], 215.0)

    def test_stationary_ball_cannot_jump_to_distant_false_positive(self) -> None:
        tracker = BallTracker(acquisition_confidence=0.30)
        for frame, x in enumerate((100.0, 101.0, 100.0, 102.0)):
            result = tracker.update(
                frame,
                [BallCandidate((x - 5.0, 95.0, x + 5.0, 105.0), 0.70)],
            )
            self.assertIsNotNone(result)

        result = tracker.update(
            4,
            [BallCandidate((195.0, 95.0, 205.0, 105.0), 0.80)],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "predicted")
        self.assertLess(abs(result.center[0] - 101.0), 5.0)

    def test_single_known_ball_cannot_jump_before_motion_is_proven(self) -> None:
        tracker = BallTracker(acquisition_confidence=0.30)
        first = tracker.update(
            0,
            [BallCandidate((100.0, 100.0, 112.0, 112.0), 0.72)],
        )
        jumped = tracker.update(
            1,
            [BallCandidate((180.0, 70.0, 192.0, 82.0), 0.46)],
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(jumped)
        self.assertEqual(jumped.source, "predicted")
        self.assertAlmostEqual(jumped.center[0], first.center[0])

    def test_direction_reversal_requires_nearby_player_contact(self) -> None:
        tracker = BallTracker(acquisition_confidence=0.30)
        for frame, x in enumerate((100.0, 110.0, 120.0, 130.0)):
            tracker.update(
                frame,
                [BallCandidate((x - 5.0, 95.0, x + 5.0, 105.0), 0.80)],
            )
        rejected = tracker.update(
            4,
            [BallCandidate((105.0, 95.0, 115.0, 105.0), 0.90)],
        )
        accepted = tracker.update(
            5,
            [BallCandidate((95.0, 95.0, 105.0, 105.0), 0.90)],
            player_footpoints=((105.0, 100.0),),
        )
        self.assertIsNotNone(rejected)
        self.assertEqual(rejected.source, "predicted")
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted.source, "detected")

    def test_strong_reacquisition_cannot_bypass_physical_speed_limit(self) -> None:
        tracker = BallTracker(
            acquisition_confidence=0.30,
            maximum_speed_pixels_per_frame=100.0,
        )
        tracker.update(0, [BallCandidate((0, 0, 10, 10), 0.40)])
        first = tracker.update(1, [BallCandidate((200, 0, 210, 10), 0.75)])
        second = tracker.update(2, [BallCandidate((202, 0, 212, 10), 0.80)])
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.source, "predicted")
        self.assertEqual(second.source, "predicted")
        self.assertLess(first.center[0], 50.0)
        self.assertLess(second.center[0], 50.0)

    def test_weak_candidates_never_become_physical_anchors(self) -> None:
        tracker = BallTracker(
            maximum_gap_frames=5,
            maximum_jump_pixels=140.0,
            maximum_speed_pixels_per_frame=100.0,
            acquisition_confidence=0.30,
            supporting_confidence=0.15,
            weak_support_radius_pixels=35.0,
        )
        tracker.update(0, [BallCandidate((0, 0, 10, 10), 0.80)])
        for frame, x in enumerate((5, 10, 15, 20, 25), start=1):
            result = tracker.update(
                frame,
                [BallCandidate((x, 0, x + 10, 10), 0.10)],
            )
            self.assertIsNotNone(result)
            self.assertEqual(result.source, "predicted")
            self.assertAlmostEqual(result.center[0], 5.0)

        # De zwakke reeks mag het anker niet richting de schoen laten kruipen.
        result = tracker.update(6, [BallCandidate((235, 0, 245, 10), 0.46)])
        self.assertIsNone(result)

    def test_weak_candidate_cannot_reacquire_after_prediction_gap(self) -> None:
        tracker = BallTracker(
            maximum_gap_frames=1,
            maximum_jump_pixels=80.0,
            acquisition_confidence=0.30,
            supporting_confidence=0.15,
        )
        tracker.update(0, [BallCandidate((95, 95, 105, 105), 0.72)])
        tracker.update(1, [])

        result = tracker.update(
            2,
            [BallCandidate((100, 95, 110, 105), 0.06)],
        )

        self.assertIsNone(result)
        self.assertEqual(len(tracker._detected_observations), 1)

    def test_support_candidate_below_acquisition_never_becomes_anchor(self) -> None:
        tracker = BallTracker(
            maximum_gap_frames=5,
            acquisition_confidence=0.50,
            supporting_confidence=0.15,
        )
        first = tracker.update(
            0,
            [BallCandidate((95, 95, 105, 105), 0.72)],
        )
        supported = tracker.update(
            1,
            [BallCandidate((107, 96, 117, 106), 0.17)],
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(supported)
        self.assertEqual(supported.source, "predicted")
        self.assertAlmostEqual(supported.center[0], first.center[0])
        self.assertNotAlmostEqual(supported.confidence, 0.17)
        self.assertEqual(len(tracker._detected_observations), 1)

    def test_support_candidate_cannot_reacquire_expired_track(self) -> None:
        tracker = BallTracker(
            maximum_gap_frames=1,
            acquisition_confidence=0.50,
            strong_reacquisition_confidence=0.55,
            supporting_confidence=0.15,
        )
        tracker.update(0, [BallCandidate((95, 95, 105, 105), 0.72)])
        tracker.update(1, [])

        result = tracker.update(
            2,
            [BallCandidate((100, 95, 110, 105), 0.17)],
        )

        self.assertIsNone(result)
        self.assertEqual(len(tracker._detected_observations), 1)

    def test_repeated_weak_trajectory_support_keeps_distant_ball_visible(self) -> None:
        tracker = BallTracker(
            maximum_gap_frames=2,
            acquisition_confidence=0.50,
            supporting_confidence=0.15,
            maximum_trajectory_support_frames=6,
        )
        first = tracker.update(
            0,
            [BallCandidate((95, 95, 105, 105), 0.72)],
        )

        supported = []
        for frame in range(1, 7):
            supported.append(
                tracker.update(
                    frame,
                    [BallCandidate((96, 95, 106, 105), 0.18)],
                )
            )

        self.assertIsNotNone(first)
        self.assertTrue(all(item is not None for item in supported))
        self.assertTrue(all(item.source == "predicted" for item in supported))
        self.assertTrue(
            all(abs(item.center[0] - first.center[0]) < 1e-6 for item in supported)
        )
        self.assertEqual(len(tracker._detected_observations), 1)

    def test_weak_trajectory_support_is_bounded_without_hard_detection(self) -> None:
        tracker = BallTracker(
            maximum_gap_frames=2,
            acquisition_confidence=0.50,
            supporting_confidence=0.15,
            maximum_trajectory_support_frames=3,
        )
        tracker.update(0, [BallCandidate((95, 95, 105, 105), 0.72)])

        visible = [
            tracker.update(
                frame,
                [BallCandidate((96, 95, 106, 105), 0.18)],
            )
            for frame in range(1, 4)
        ]
        expired = tracker.update(
            4,
            [BallCandidate((96, 95, 106, 105), 0.18)],
        )

        self.assertTrue(all(item is not None for item in visible))
        self.assertIsNone(expired)
        self.assertEqual(len(tracker._detected_observations), 1)

    def test_player_activity_briefly_supports_existing_trajectory(self) -> None:
        tracker = BallTracker(
            maximum_gap_frames=1,
            acquisition_confidence=0.50,
            maximum_player_activity_support_frames=3,
            player_activity_radius_pixels=90.0,
            minimum_activity_players=2,
        )
        tracker.update(0, [BallCandidate((95, 95, 105, 105), 0.72)])

        supported = [
            tracker.update(
                frame,
                [],
                player_footpoints=((170.0, 100.0), (100.0, 170.0)),
            )
            for frame in range(1, 4)
        ]
        expired = tracker.update(
            4,
            [],
            player_footpoints=((170.0, 100.0), (100.0, 170.0)),
        )

        self.assertTrue(all(item is not None for item in supported))
        self.assertTrue(all(item.source == "predicted" for item in supported))
        self.assertIsNone(expired)
        self.assertEqual(len(tracker._detected_observations), 1)

    def test_player_activity_cannot_start_ball_track(self) -> None:
        tracker = BallTracker(
            maximum_player_activity_support_frames=3,
            player_activity_radius_pixels=90.0,
            minimum_activity_players=2,
        )

        result = tracker.update(
            0,
            [],
            player_footpoints=((100.0, 100.0), (130.0, 110.0)),
        )

        self.assertIsNone(result)
        self.assertEqual(tracker.observations, ())

    def test_distant_player_activity_does_not_extend_ball_track(self) -> None:
        tracker = BallTracker(
            maximum_gap_frames=1,
            acquisition_confidence=0.50,
            maximum_player_activity_support_frames=3,
            player_activity_radius_pixels=90.0,
            minimum_activity_players=2,
        )
        tracker.update(0, [BallCandidate((95, 95, 105, 105), 0.72)])
        tracker.update(1, [])

        result = tracker.update(
            2,
            [],
            player_footpoints=((500.0, 500.0), (540.0, 510.0)),
        )

        self.assertIsNone(result)
        self.assertEqual(len(tracker._detected_observations), 1)

    def test_nearby_detection_beats_distant_strong_false_positive(self) -> None:
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
        self.assertIsNotNone(pending)
        self.assertEqual(pending.source, "detected")
        self.assertAlmostEqual(pending.center[0], 17.0)
        selected = tracker.update(
            2,
            [BallCandidate((910, 302, 920, 312), 0.70)],
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "predicted")
        self.assertLess(selected.center[0], 50.0)

    def test_unconfirmed_distant_reacquisition_is_discarded(self) -> None:
        tracker = BallTracker(
            maximum_jump_pixels=80.0,
            acquisition_confidence=0.30,
        )
        tracker.update(0, [BallCandidate((10, 10, 20, 20), 0.40)])
        first = tracker.update(1, [BallCandidate((900, 300, 910, 310), 0.70)])
        self.assertIsNotNone(first)
        self.assertEqual(first.source, "predicted")
        self.assertAlmostEqual(first.center[0], 15.0)
        result = tracker.update(2, [BallCandidate((300, 100, 310, 110), 0.72)])
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "predicted")
        self.assertAlmostEqual(result.center[0], 15.0)

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
        candidate = BallCandidate((45, 98, 55, 108), 0.80)
        result = exclude_candidates_inside_people([candidate], [(20, 20, 80, 100)])
        self.assertEqual(result, [candidate])

    def test_rejects_shoe_candidate_inside_bottom_of_person_box(self) -> None:
        candidate = BallCandidate((45, 90, 55, 99), 0.80)
        result = exclude_candidates_inside_people([candidate], [(20, 20, 80, 100)])
        self.assertEqual(result, [])

    def test_keeps_ball_touching_outside_of_person_box(self) -> None:
        candidate = BallCandidate((75, 92, 89, 106), 0.80)
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

    def test_holds_stationary_ball_across_long_detection_gap(self) -> None:
        observations = (
            BallObservation(0, (200.0, 300.0), (190.0, 290.0, 210.0, 310.0), 0.72, "detected"),
            BallObservation(180, (205.0, 302.0), (195.0, 292.0, 215.0, 312.0), 0.81, "detected"),
        )
        result = hold_stationary_detected_gaps(
            observations,
            maximum_gap_frames=240,
        )
        by_frame = {item.frame_number: item for item in result}
        self.assertEqual(len(result), 181)
        self.assertEqual(by_frame[90].source, "stationary_hold")
        self.assertAlmostEqual(by_frame[90].center[0], 202.5)

    def test_does_not_hold_ball_when_endpoints_moved(self) -> None:
        observations = (
            BallObservation(0, (200.0, 300.0), (190.0, 290.0, 210.0, 310.0), 0.72, "detected"),
            BallObservation(180, (500.0, 300.0), (490.0, 290.0, 510.0, 310.0), 0.81, "detected"),
        )
        result = hold_stationary_detected_gaps(
            observations,
            maximum_gap_frames=240,
        )
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
