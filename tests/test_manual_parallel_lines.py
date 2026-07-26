import unittest

from football_ai.calibration.manual_midfield_line import ManualMidfieldLine
from football_ai.calibration.manual_parallel_lines import (
    ManualParallelLine,
    ManualParallelLineReference,
)


class ManualParallelLineTests(unittest.TestCase):
    def test_reference_round_trip(self):
        points = ((10.0, 20.0), (30.0, 21.0), (50.0, 22.0), (70.0, 23.0), (90.0, 24.0))
        midfield = ManualMidfieldLine.fit("match.mp4", 1, 0.1, points)
        reference = ManualParallelLineReference(
            "match.mp4",
            (
                ManualParallelLine.from_midfield(midfield),
                ManualParallelLine.fit("goal_area_5m", 2, 0.2, points),
                ManualParallelLine.fit("penalty_area_16m", 3, 0.3, points),
            ),
        )

        restored = ManualParallelLineReference.from_dict(reference.to_dict())

        self.assertEqual(restored, reference)
        self.assertEqual(restored.to_dict()["world_relation"], "parallel")

    def test_requires_all_three_lines_in_semantic_order(self):
        points = ((10.0, 20.0), (30.0, 21.0), (50.0, 22.0), (70.0, 23.0), (90.0, 24.0))
        with self.assertRaisesRegex(ValueError, "middenlijn"):
            ManualParallelLineReference(
                "match.mp4",
                (ManualParallelLine.fit("goal_area_5m", 2, 0.2, points),),
            )


if __name__ == "__main__":
    unittest.main()
