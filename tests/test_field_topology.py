import unittest

from football_ai.calibration.field_topology import (
    CORNER_CYCLE,
    boundary_between,
    boundary_corner_pairs,
    ground_corners,
)


class FieldTopologyTests(unittest.TestCase):
    def test_canonical_cycle_matches_confirmed_corner_order(self):
        self.assertEqual(
            CORNER_CYCLE,
            ("linksachter", "rechtsachter", "rechtsvoor", "linksvoor"),
        )
        pairs = tuple((first, second) for _name, first, second in boundary_corner_pairs())
        self.assertEqual(
            pairs,
            tuple(zip(CORNER_CYCLE, (*CORNER_CYCLE[1:], CORNER_CYCLE[0]))),
        )

    def test_boundaries_are_bidirectional_but_never_diagonal(self):
        self.assertEqual(boundary_between("rechtsachter", "rechtsvoor"), "end_line_b")
        self.assertEqual(boundary_between("rechtsvoor", "rechtsachter"), "end_line_b")
        with self.assertRaisesRegex(ValueError, "Niet-aangrenzende"):
            boundary_between("linksachter", "rechtsvoor")

    def test_metric_corner_mapping_preserves_names(self):
        self.assertEqual(
            ground_corners(64.0, 42.5),
            {
                "linksachter": (0.0, 0.0),
                "rechtsachter": (64.0, 0.0),
                "rechtsvoor": (64.0, 42.5),
                "linksvoor": (0.0, 42.5),
            },
        )


if __name__ == "__main__":
    unittest.main()
