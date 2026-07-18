from __future__ import annotations

import cv2
import numpy as np

from football_ai.pitch.calibration_model import PitchCalibration


class PitchFilter:
    """
    Controleert of de grondcontactpunten van een detectie binnen
    het gekalibreerde speelveld liggen.

    Alleen punten aan de onderzijde van de bounding box worden
    geprojecteerd. De homografie beschrijft namelijk het grondvlak;
    bovenhoeken van een speler mogen niet als veldpunten worden
    geïnterpreteerd.
    """

    def __init__(
        self,
        calibration: PitchCalibration,
        foot_margin_m: float = 0.5,
        support_margin_m: float = 1.0,
        minimum_support_ratio: float = 2.0 / 3.0,
    ) -> None:
        if foot_margin_m < 0:
            raise ValueError("foot_margin_m mag niet negatief zijn.")

        if support_margin_m < 0:
            raise ValueError(
                "support_margin_m mag niet negatief zijn."
            )

        if not 0.0 <= minimum_support_ratio <= 1.0:
            raise ValueError(
                "minimum_support_ratio moet tussen 0 en 1 liggen."
            )

        self.calibration = calibration
        self.foot_margin_m = foot_margin_m
        self.support_margin_m = support_margin_m
        self.minimum_support_ratio = minimum_support_ratio

    def accept(
        self,
        bounding_box: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        ground_points = self._ground_contact_points(
            bounding_box=bounding_box,
        )

        pitch_points = self.project_image_points(
            image_points=ground_points,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        if pitch_points is None:
            return False

        # Het centrale voetpunt moet dichtbij of binnen het echte
        # speelveld liggen. Dit verwijdert staf en publiek die duidelijk
        # buiten de lijn staan.
        center_foot = pitch_points[1]

        if not self._inside_pitch(
            point=center_foot,
            margin_m=self.foot_margin_m,
        ):
            return False

        # Minimaal twee van de drie punten aan de onderzijde moeten
        # binnen de ruimere steunzone vallen. Dit houdt spelers die
        # deels over de lijn bewegen beter vast.
        support_results = [
            self._inside_pitch(
                point=point,
                margin_m=self.support_margin_m,
            )
            for point in pitch_points
        ]

        support_ratio = (
            sum(support_results) / len(support_results)
        )

        return support_ratio >= self.minimum_support_ratio

    def project_image_points(
        self,
        image_points: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray | None:
        if frame_width <= 0 or frame_height <= 0:
            return None

        points = np.asarray(
            image_points,
            dtype=np.float32,
        )

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                "image_points moet de vorm (n, 2) hebben."
            )

        scale_x = (
            self.calibration.frame_width / frame_width
        )
        scale_y = (
            self.calibration.frame_height / frame_height
        )

        scaled_points = points.copy()
        scaled_points[:, 0] *= scale_x
        scaled_points[:, 1] *= scale_y

        transformed = cv2.perspectiveTransform(
            scaled_points.reshape(-1, 1, 2),
            self.calibration.image_to_pitch_matrix,
        ).reshape(-1, 2)

        if not np.all(np.isfinite(transformed)):
            return None

        return transformed

    def _ground_contact_points(
        self,
        bounding_box: np.ndarray,
    ) -> np.ndarray:
        x1, _y1, x2, y2 = bounding_box.astype(float)
        box_width = x2 - x1

        # Niet helemaal op de uiterste bbox-hoeken samplen:
        # detectieboxen bevatten vaak wat achtergrond.
        left_x = x1 + 0.25 * box_width
        center_x = (x1 + x2) / 2.0
        right_x = x2 - 0.25 * box_width

        return np.array(
            [
                [left_x, y2],
                [center_x, y2],
                [right_x, y2],
            ],
            dtype=np.float32,
        )

    def _inside_pitch(
        self,
        point: np.ndarray,
        margin_m: float,
    ) -> bool:
        x = float(point[0])
        y = float(point[1])
        profile = self.calibration.profile

        return (
            -margin_m <= x <= profile.width_m + margin_m
            and -margin_m <= y <= profile.length_m + margin_m
        )
