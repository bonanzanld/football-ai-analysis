import cv2
import numpy as np

from football_ai.tracking.online_camera_motion import (
    OnlineCameraMotion,
    transform_box,
    transform_point,
)


def _textured_frame() -> np.ndarray:
    frame = np.zeros((480, 800, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for x, y in rng.integers([20, 70], [780, 460], size=(300, 2)):
        cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 255), -1)
    cv2.rectangle(frame, (80, 160), (330, 390), (80, 180, 80), 3)
    return frame


def test_translation_is_mapped_back_to_reference_frame() -> None:
    reference = _textured_frame()
    dx, dy = 18.0, -7.0
    shifted = cv2.warpAffine(
        reference,
        np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
        (reference.shape[1], reference.shape[0]),
    )
    motion = OnlineCameraMotion()
    motion.update(reference)
    current_to_reference = motion.update(shifted)

    mapped = transform_point((400.0 + dx, 300.0 + dy), current_to_reference)
    assert np.allclose(mapped, (400.0, 300.0), atol=2.5)
    assert motion.accepted_updates == 1


def test_blank_frames_keep_identity_transform() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    motion = OnlineCameraMotion()
    first = motion.update(frame)
    second = motion.update(frame)

    assert np.allclose(first, np.eye(3))
    assert np.allclose(second, np.eye(3))
    assert motion.rejected_updates == 1


def test_rejected_frame_does_not_replace_last_good_reference() -> None:
    reference = _textured_frame()
    dx, dy = 24.0, 5.0
    shifted = cv2.warpAffine(
        reference,
        np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
        (reference.shape[1], reference.shape[0]),
    )
    motion = OnlineCameraMotion()
    motion.update(reference)
    motion.update(np.zeros_like(reference))
    current_to_reference = motion.update(shifted)

    mapped = transform_point((400.0 + dx, 300.0 + dy), current_to_reference)
    assert np.allclose(mapped, (400.0, 300.0), atol=2.5)
    assert motion.accepted_updates == 1
    assert motion.rejected_updates == 1


def test_transform_box_uses_all_four_corners() -> None:
    transform = np.array(
        [[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    transformed = transform_box((10.0, 20.0, 30.0, 40.0), transform)

    assert np.allclose(transformed, (-30.0, 30.0, 10.0, 70.0))
