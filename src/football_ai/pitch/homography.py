from __future__ import annotations

import cv2
import numpy as np

from football_ai.pitch.pitch_model import (
    PitchCalibration,
)


class PitchHomography:
    def __init__(
        self,
        calibration: PitchCalibration,
    ) -> None:
        self.calibration = calibration

    def image_to_pitch(
        self,
        image_points: np.ndarray,
    ) -> np.ndarray:
        """
        Zet beeldcoördinaten om naar veldcoördinaten in meters.
        """
        points = self._prepare_points(
            image_points
        )

        transformed = cv2.perspectiveTransform(
            points,
            self.calibration.image_to_pitch_matrix,
        )

        return transformed.reshape(-1, 2)

    def pitch_to_image(
        self,
        pitch_points: np.ndarray,
    ) -> np.ndarray:
        """
        Zet veldcoördinaten in meters om naar beeldcoördinaten.
        """
        points = self._prepare_points(
            pitch_points
        )

        transformed = cv2.perspectiveTransform(
            points,
            self.calibration.pitch_to_image_matrix,
        )

        return transformed.reshape(-1, 2)

    def is_image_point_inside_pitch(
        self,
        image_point: tuple[float, float],
        margin_m: float = 0.0,
    ) -> bool:
        pitch_point = self.image_to_pitch(
            np.array(
                [image_point],
                dtype=np.float32,
            )
        )[0]

        x, y = pitch_point

        profile = self.calibration.profile

        return (
            -margin_m
            <= x
            <= profile.width_m + margin_m
            and -margin_m
            <= y
            <= profile.length_m + margin_m
        )

    @staticmethod
    def _prepare_points(
        points: np.ndarray,
    ) -> np.ndarray:
        converted = np.asarray(
            points,
            dtype=np.float32,
        )

        if converted.ndim != 2:
            raise ValueError(
                "Punten moeten de vorm (n, 2) hebben."
            )

        if converted.shape[1] != 2:
            raise ValueError(
                "Elk punt moet een x- en y-coördinaat hebben."
            )

        return converted.reshape(-1, 1, 2)