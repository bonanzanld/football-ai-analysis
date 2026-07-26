import unittest

import numpy as np

from football_ai.calibration.manual_perspective_reference import (
    ManualPerspectiveView,
    ManualReferenceLine,
    PerspectiveDirection,
    assess_three_view_consistency,
    automatically_classify_line_directions,
    assess_global_readiness,
)


class ManualPerspectiveReferenceTests(unittest.TestCase):
    @staticmethod
    def _line(direction, start, vanishing):
        start = np.asarray(start, dtype=np.float64)
        vanishing = np.asarray(vanishing, dtype=np.float64)
        points = tuple(tuple(start + fraction * (vanishing - start)) for fraction in (0.0, 0.2, 0.4))
        return ManualReferenceLine(direction, points)

    def test_three_point_lines_recover_vanishing_points_and_horizon(self) -> None:
        first, second = (1600.0, 300.0), (-500.0, 250.0)
        view = ManualPerspectiveView(
            "center", 100, 5.0,
            (
                self._line(PerspectiveDirection.BETWEEN_GOALS, (100.0, 100.0), first),
                self._line(PerspectiveDirection.BETWEEN_GOALS, (100.0, 500.0), first),
                self._line(PerspectiveDirection.ALONG_END_LINES, (300.0, 100.0), second),
                self._line(PerspectiveDirection.ALONG_END_LINES, (900.0, 600.0), second),
            ),
        )
        np.testing.assert_allclose(view.vanishing_point(PerspectiveDirection.BETWEEN_GOALS), first, atol=1e-6)
        np.testing.assert_allclose(view.vanishing_point(PerspectiveDirection.ALONG_END_LINES), second, atol=1e-6)
        horizon = np.asarray(view.horizon())
        self.assertAlmostEqual(float(horizon @ np.asarray((*first, 1.0))), 0.0, places=5)
        self.assertAlmostEqual(float(horizon @ np.asarray((*second, 1.0))), 0.0, places=5)

    def test_rejects_inconsistent_camera_geometry_across_three_views(self) -> None:
        def view(label, first, second):
            return ManualPerspectiveView(
                label, 1, 0.0,
                (
                    self._line(PerspectiveDirection.BETWEEN_GOALS, (100.0, 100.0), first),
                    self._line(PerspectiveDirection.BETWEEN_GOALS, (100.0, 500.0), first),
                    self._line(PerspectiveDirection.ALONG_END_LINES, (300.0, 100.0), second),
                    self._line(PerspectiveDirection.ALONG_END_LINES, (900.0, 600.0), second),
                ),
            )
        views = (
            view("left_goal", (1600.0, 300.0), (-500.0, 250.0)),
            view("center", (1600.0, 300.0), (-500.0, 250.0)),
            view("right_goal", (800.0, 300.0), (500.0, 250.0)),
        )
        valid, _focals, _reason = assess_three_view_consistency(views, (1280, 720))
        self.assertFalse(valid)

    def test_automatically_splits_unlabelled_lines_into_two_families(self) -> None:
        first, second = (1600.0, 300.0), (-500.0, 250.0)
        lines = tuple(
            self._line(PerspectiveDirection.UNKNOWN, start, first)
            for start in ((100.0, 100.0), (100.0, 300.0), (100.0, 500.0))
        ) + tuple(
            self._line(PerspectiveDirection.UNKNOWN, start, second)
            for start in ((300.0, 100.0), (600.0, 300.0), (900.0, 600.0))
        )
        classified = automatically_classify_line_directions(lines, (1280, 720))
        counts = {
            direction: sum(item.direction is direction for item in classified)
            for direction in (PerspectiveDirection.BETWEEN_GOALS, PerspectiveDirection.ALONG_END_LINES)
        }
        self.assertEqual(sorted(counts.values()), [3, 3])

    def test_two_unlabelled_lines_are_valid_partial_evidence(self) -> None:
        lines = (
            self._line(PerspectiveDirection.UNKNOWN, (100.0, 100.0), (1600.0, 300.0)),
            self._line(PerspectiveDirection.UNKNOWN, (300.0, 500.0), (-500.0, 250.0)),
        )
        view = ManualPerspectiveView("left_goal", 1, 0.0, lines)
        self.assertFalse(view.perspective_complete)
        data = view.to_dict()
        self.assertIsNone(data["horizon"])
        self.assertIsNone(data["vanishing_points"])

    def test_one_complete_and_two_partial_views_are_globally_ready(self) -> None:
        first, second = (1600.0, 300.0), (-500.0, 250.0)
        complete = ManualPerspectiveView(
            "right_goal", 3, 2.0,
            (
                self._line(PerspectiveDirection.BETWEEN_GOALS, (100.0, 100.0), first),
                self._line(PerspectiveDirection.BETWEEN_GOALS, (100.0, 500.0), first),
                self._line(PerspectiveDirection.ALONG_END_LINES, (300.0, 100.0), second),
                self._line(PerspectiveDirection.ALONG_END_LINES, (900.0, 600.0), second),
            ),
        )
        partial_a = ManualPerspectiveView("left_goal", 1, 0.0, complete.lines[:3])
        partial_b = ManualPerspectiveView("center", 2, 1.0, complete.lines[:2])
        ready, _reason = assess_global_readiness((partial_a, partial_b, complete))
        self.assertTrue(ready)


if __name__ == "__main__":
    unittest.main()
