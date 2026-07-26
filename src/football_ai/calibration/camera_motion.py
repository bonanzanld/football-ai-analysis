from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraMotionKeyframe:
    frame_number: int
    frame_to_panorama_matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = _validate_affine_matrix(self.frame_to_panorama_matrix)
        object.__setattr__(self, "frame_to_panorama_matrix", matrix)

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "frame_to_panorama_matrix": self.frame_to_panorama_matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraMotionKeyframe":
        return cls(
            frame_number=int(data["frame_number"]),
            frame_to_panorama_matrix=np.asarray(
                data["frame_to_panorama_matrix"], dtype=np.float64
            ),
        )


@dataclass(frozen=True)
class CameraMotionTrajectory:
    panorama_to_pitch_matrix: np.ndarray
    keyframes: tuple[CameraMotionKeyframe, ...]

    def __post_init__(self) -> None:
        panorama_to_pitch = np.asarray(
            self.panorama_to_pitch_matrix, dtype=np.float64
        )
        if panorama_to_pitch.shape != (3, 3):
            raise ValueError("panorama_to_pitch_matrix moet een 3x3-matrix zijn.")
        if not np.all(np.isfinite(panorama_to_pitch)):
            raise ValueError("panorama_to_pitch_matrix bevat ongeldige waarden.")
        ordered = tuple(sorted(self.keyframes, key=lambda item: item.frame_number))
        if not ordered:
            raise ValueError("Een cameratraject vereist minimaal één keyframe.")
        if len({item.frame_number for item in ordered}) != len(ordered):
            raise ValueError("Camerakeyframes moeten unieke framenummers hebben.")
        object.__setattr__(self, "panorama_to_pitch_matrix", panorama_to_pitch)
        object.__setattr__(self, "keyframes", ordered)

    def image_to_pitch_for_frame(self, frame_number: int) -> np.ndarray:
        frame_to_panorama = self.frame_to_panorama_for_frame(frame_number)
        image_to_pitch = self.panorama_to_pitch_matrix @ frame_to_panorama
        if abs(image_to_pitch[2, 2]) < 1e-12:
            return self.nearest_image_to_pitch(frame_number)
        image_to_pitch /= image_to_pitch[2, 2]
        if not np.all(np.isfinite(image_to_pitch)):
            return self.nearest_image_to_pitch(frame_number)
        return image_to_pitch

    def frame_to_panorama_for_frame(self, frame_number: int) -> np.ndarray:
        if frame_number <= self.keyframes[0].frame_number:
            return self.keyframes[0].frame_to_panorama_matrix.copy()
        if frame_number >= self.keyframes[-1].frame_number:
            return self.keyframes[-1].frame_to_panorama_matrix.copy()

        for first, second in zip(self.keyframes, self.keyframes[1:]):
            if first.frame_number <= frame_number <= second.frame_number:
                span = second.frame_number - first.frame_number
                alpha = (frame_number - first.frame_number) / span
                interpolated = (
                    (1.0 - alpha) * first.frame_to_panorama_matrix
                    + alpha * second.frame_to_panorama_matrix
                )
                interpolated[2] = (0.0, 0.0, 1.0)
                try:
                    return _validate_affine_matrix(interpolated)
                except ValueError:
                    return (
                        first.frame_to_panorama_matrix.copy()
                        if alpha < 0.5
                        else second.frame_to_panorama_matrix.copy()
                    )
        return self.keyframes[-1].frame_to_panorama_matrix.copy()

    def nearest_image_to_pitch(self, frame_number: int) -> np.ndarray:
        nearest = min(
            self.keyframes,
            key=lambda item: abs(item.frame_number - frame_number),
        )
        matrix = self.panorama_to_pitch_matrix @ nearest.frame_to_panorama_matrix
        return matrix / matrix[2, 2]

    def to_dict(self) -> dict:
        return {
            "panorama_to_pitch_matrix": self.panorama_to_pitch_matrix.tolist(),
            "keyframes": [item.to_dict() for item in self.keyframes],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraMotionTrajectory":
        return cls(
            panorama_to_pitch_matrix=np.asarray(
                data["panorama_to_pitch_matrix"], dtype=np.float64
            ),
            keyframes=tuple(
                CameraMotionKeyframe.from_dict(item)
                for item in data["keyframes"]
            ),
        )


def _validate_affine_matrix(matrix: np.ndarray) -> np.ndarray:
    converted = np.asarray(matrix, dtype=np.float64).copy()
    if converted.shape != (3, 3):
        raise ValueError("Cameratransformatie moet een 3x3-matrix zijn.")
    if not np.all(np.isfinite(converted)):
        raise ValueError("Cameratransformatie bevat ongeldige waarden.")
    if not np.allclose(converted[2], (0.0, 0.0, 1.0), atol=1e-7):
        raise ValueError("Cameratransformatie moet affine zijn.")
    determinant = float(np.linalg.det(converted[:2, :2]))
    if determinant <= 1e-6:
        raise ValueError("Cameratransformatie heeft een ongeldige determinant.")
    return converted
