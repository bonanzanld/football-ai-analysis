from pathlib import Path

import cv2
import numpy as np

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.bootstrap.goal_seed import GoalSeed
from football_ai.calibration.reference_3d import create_field_reference_3d
from football_ai.calibration.reference_observation import CameraViewObservations, ReferenceObservation2D
from football_ai.calibration.camera_projection_3d import CameraProjection3D, CameraProjectionEstimate
from football_ai.calibration.reference_observation_app import ReferenceObservationApp


def test_resume_preserves_existing_observations(tmp_path: Path, monkeypatch):
    video = tmp_path / "frame.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 1.0, (64, 48))
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    seed = GoalSeed("A", 0, 0.0, 1, 0.0, (20.0, 20.0), (30.0, 20.0), 5.0, 0.5)
    existing = CameraViewObservations(
        0,
        1,
        (ReferenceObservation2D("goal_a_rear_top", (20.0, 10.0)),),
    )

    app = ReferenceObservationApp(
        video,
        seed,
        create_field_reference_3d(create_detection_profile("8v8")),
        existing_view=existing,
        requested_landmarks=("corner_a_rear",),
    )

    assert app.requested == ("corner_a_rear",)
    assert "goal_a_rear_top" in {item.landmark_id for item in app.observations}


def test_camera_selection_prefers_lowest_error_after_direction_gate(tmp_path: Path, monkeypatch):
    video = tmp_path / "frame.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 1.0, (64, 48))
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    seed = GoalSeed(
        "A", 0, 0.0, 1, 0.0, (20.0, 30.0), (30.0, 30.0), 5.0, 0.5,
        rear_corner=(10.0, 30.0), front_corner=(40.0, 30.0),
        rear_sideline_support=(12.0, 20.0), front_sideline_support=(42.0, 20.0),
    )
    app = ReferenceObservationApp(
        video,
        seed,
        create_field_reference_3d(create_detection_profile("8v8")),
    )
    app.observations.extend((
        ReferenceObservation2D("goal_a_rear_top", (20.0, 20.0)),
        ReferenceObservation2D("goal_a_front_top", (30.0, 20.0)),
    ))
    projection = CameraProjection3D(np.asarray(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)), dtype=float))
    estimates = iter((
        CameraProjectionEstimate(projection, (6.0,), 6.0, 6.0),
        CameraProjectionEstimate(projection, (2.0,), 2.0, 2.0),
    ))
    monkeypatch.setattr("football_ai.calibration.reference_observation_app.np.linspace", lambda *_args: (1.0, 2.0))
    monkeypatch.setattr("football_ai.calibration.reference_observation_app.estimate_camera_from_goal_plane", lambda *_args, **_kwargs: next(estimates))
    monkeypatch.setattr("football_ai.calibration.reference_observation_app.orient_projection_toward_field", lambda _r, estimate, _s: estimate)
    monkeypatch.setattr("football_ai.calibration.reference_observation_app.field_direction_score", lambda *_args: 0.9)
    monkeypatch.setattr("football_ai.calibration.reference_observation_app.validate_projected_pitch_geometry", lambda *_args, **_kwargs: type("Geometry", (), {"errors": ()})())

    result = app._build_result()

    assert result.estimate is not None
    assert result.estimate.rms_error_px == 2.0
