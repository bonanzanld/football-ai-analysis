import unittest

from football_ai.analysis.entity_timeline import TimelineEntity
from football_ai.tracking.entity_corrections import EntityRole, TeamAssignment
from football_ai.visualization.tactical_map import (
    CameraRelativeProjector,
    GoalkeeperAnchoredProjector,
)


class CameraRelativeProjectorTests(unittest.TestCase):
    def test_clamps_points_to_map(self) -> None:
        projector = CameraRelativeProjector()
        self.assertEqual((projector.project((-20, -50), (1280, 720)).x, projector.project((-20, -50), (1280, 720)).y), (0.0, 0.0))
        self.assertEqual((projector.project((1400, 900), (1280, 720)).x, projector.project((1400, 900), (1280, 720)).y), (1.0, 1.0))

    def test_preserves_horizontal_order_and_depth(self) -> None:
        projector = CameraRelativeProjector()
        left = projector.project((200, 500), (1280, 720))
        right = projector.project((900, 500), (1280, 720))
        far = projector.project((600, 320), (1280, 720))
        near = projector.project((600, 650), (1280, 720))
        self.assertLess(left.x, right.x)
        self.assertLess(far.y, near.y)

    def test_is_explicitly_non_metric(self) -> None:
        projector = CameraRelativeProjector()
        self.assertFalse(projector.metric)
        self.assertEqual(projector.name, "camera-relatief")


class GoalkeeperAnchoredProjectorTests(unittest.TestCase):
    @staticmethod
    def _keeper(x: float, y: float, track_id: int = 1) -> TimelineEntity:
        return TimelineEntity(
            frame_number=0,
            track_id=track_id,
            identity_id=track_id,
            label=f"Keeper {track_id}",
            role=EntityRole.GOALKEEPER,
            team=TeamAssignment.TEAM_A,
            box=(x - 10, y - 40, x + 10, y),
            footpoint=(x, y),
        )

    def test_falls_back_without_visible_goalkeeper(self) -> None:
        projector = GoalkeeperAnchoredProjector()
        expected = CameraRelativeProjector(depth_exponent=1.225).project((640, 500), (1280, 720))
        actual = projector.project((640, 500), (1280, 720))
        self.assertAlmostEqual(actual.x, expected.x)
        self.assertAlmostEqual(actual.y, expected.y)
        self.assertFalse(projector.anchored)

    def test_left_goalkeeper_is_placed_in_front_of_left_goal(self) -> None:
        projector = GoalkeeperAnchoredProjector()
        keeper = self._keeper(260, 430)
        projector.update([keeper], (1280, 720))
        point = projector.project(keeper.footpoint, (1280, 720))
        self.assertAlmostEqual(point.x, projector.goal_offset)
        self.assertAlmostEqual(point.y, projector.goal_center_y)
        self.assertTrue(projector.anchored)

    def test_anchor_warp_keeps_horizontal_and_depth_order(self) -> None:
        projector = GoalkeeperAnchoredProjector()
        projector.update([self._keeper(260, 430)], (1280, 720))
        left = projector.project((100, 500), (1280, 720))
        middle = projector.project((600, 500), (1280, 720))
        right = projector.project((1100, 500), (1280, 720))
        far = projector.project((600, 330), (1280, 720))
        near = projector.project((600, 650), (1280, 720))
        self.assertLess(left.x, middle.x)
        self.assertLess(middle.x, right.x)
        self.assertLess(far.y, near.y)


if __name__ == "__main__":
    unittest.main()
