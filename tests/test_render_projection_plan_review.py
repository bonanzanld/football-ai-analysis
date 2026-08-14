import numpy as np

from football_ai.calibration.camera_projection_3d import CameraProjection3D
from tools.render_projection_plan_review import _draw_player_footpoints


def test_draw_player_footpoints_counts_inside_and_outside():
    frame = np.zeros((120, 140, 3), dtype=np.uint8)
    projection = CameraProjection3D(
        np.asarray(
            ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        )
    )

    acceptable, outside = _draw_player_footpoints(
        frame,
        np.asarray(((5, 0, 15, 10), (95, 60, 105, 80)), dtype=float),
        projection,
        64.0,
        42.5,
        2.0,
    )

    assert (acceptable, outside) == (1, 1)
    assert np.any(frame)
