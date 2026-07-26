import unittest

from football_ai.calibration.bootstrap.field_anchor_bank import build_field_anchor_bank
from football_ai.calibration.bootstrap.goal_seed import GoalSeed
from football_ai.calibration.bootstrap.sideline_anchor import SidelineAnchor


class FieldAnchorBankTests(unittest.TestCase):
    def test_keeps_unobserved_intermediate_boundary_open(self) -> None:
        goals = (
            GoalSeed("A", 10, 1.0, 1, 0.0, (100, 100), (100, 140), 5.0, 0.2,
                     rear_corner=(100, 40), front_corner=(100, 300),
                     rear_sideline_support=(300, 80), front_sideline_support=(300, 260)),
            GoalSeed("B", 20, 2.0, 2, 1.0, (540, 120), (540, 160), 5.0, 0.2,
                     rear_corner=(540, 60), front_corner=(540, 320),
                     rear_sideline_support=(300, 100), front_sideline_support=(300, 280)),
        )
        middle = SidelineAnchor(3, 0.5, 15, 1.5, (320, 90), None)
        anchors = build_field_anchor_bank(goals, (middle,), 42.5, (640, 360))
        selected = next(item for item in anchors if item.anchor_id == "stand-3")
        self.assertIsNotNone(selected.rear_line)
        self.assertIsNone(selected.front_line)
        self.assertIsNone(selected.backline)


if __name__ == "__main__":
    unittest.main()
