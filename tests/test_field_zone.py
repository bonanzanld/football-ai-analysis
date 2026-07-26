import unittest

from football_ai.calibration.field_zone import FieldZone, classify_field_position


class FieldZoneTests(unittest.TestCase):
    def test_classifies_inside_edge_and_outside(self) -> None:
        self.assertEqual(classify_field_position((20.0, 15.0), 64.0, 42.5), FieldZone.INSIDE)
        self.assertEqual(classify_field_position((0.4, 20.0), 64.0, 42.5), FieldZone.EDGE)
        self.assertEqual(classify_field_position((-1.0, 20.0), 64.0, 42.5), FieldZone.EDGE)
        self.assertEqual(classify_field_position((-2.0, 20.0), 64.0, 42.5), FieldZone.OUTSIDE)

    def test_works_with_every_supported_profile_size(self) -> None:
        for length, width in ((42.5, 30.0), (64.0, 42.5), (105.0, 68.0)):
            self.assertEqual(classify_field_position((length / 2.0, width / 2.0), length, width), FieldZone.INSIDE)


if __name__ == "__main__":
    unittest.main()
