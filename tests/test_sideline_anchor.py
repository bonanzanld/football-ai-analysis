import unittest

from football_ai.calibration.bootstrap.sideline_anchor import (
    SidelineAnchor,
    select_intermediate_camera_states,
)


class SidelineAnchorTests(unittest.TestCase):
    def test_round_trip_preserves_two_ground_points(self) -> None:
        anchor = SidelineAnchor(3, 0.5, 900, 30.0, (10.0, 20.0), (30.0, 40.0))
        self.assertEqual(SidelineAnchor.from_dict(anchor.to_dict()), anchor)

    def test_round_trip_preserves_an_invisible_sideline(self) -> None:
        anchor = SidelineAnchor(4, 0.7, 1200, 40.0, (10.0, 20.0), None)
        self.assertEqual(SidelineAnchor.from_dict(anchor.to_dict()), anchor)

    def test_selects_non_goal_states_in_view_order(self) -> None:
        report = {
            "camera_states": [
                {"camera_state": 1, "view_position": 0.6},
                {"camera_state": 2, "view_position": 0.9},
                {"camera_state": 3, "view_position": 0.2},
            ]
        }
        selected = select_intermediate_camera_states(report, {2})
        self.assertEqual([item["camera_state"] for item in selected], [3, 1])


if __name__ == "__main__":
    unittest.main()
