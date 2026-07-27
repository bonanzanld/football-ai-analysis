import unittest

from football_ai.visualization.tactical_map import CameraRelativeProjector


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


if __name__ == "__main__":
    unittest.main()
