from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import cv2
import numpy as np

from football_ai.pitch.calibration_model import PitchCalibration


@dataclass(frozen=True)
class FieldPosition:
    """
    Positie in meters op het gekalibreerde voetbalveld.
    """

    x: float
    y: float
    is_inside_pitch: bool

    @property
    def point(self) -> tuple[float, float]:
        return self.x, self.y


class FieldProjector:
    """
    Zet beeldcoördinaten om naar veldcoördinaten in meters.

    Voor spelers gebruiken we het voetpunt van de bounding box:
    het midden van de onderzijde van de detectie.

    Ongeldige of numeriek onrealistische projecties worden afgewezen
    en als None teruggegeven. Hierdoor komen extreme coördinaten niet
    terecht in TrackState en de verdere evaluatiepipeline.
    """

    def __init__(
        self,
        calibration: PitchCalibration,
        pitch_margin_m: float = 0.0,
        maximum_outside_distance_m: float = 30.0,
        minimum_homogeneous_scale: float = 1e-6,
    ) -> None:
        if pitch_margin_m < 0.0:
            raise ValueError(
                "pitch_margin_m mag niet negatief zijn."
            )

        if maximum_outside_distance_m < 0.0:
            raise ValueError(
                "maximum_outside_distance_m mag niet negatief zijn."
            )

        if minimum_homogeneous_scale <= 0.0:
            raise ValueError(
                "minimum_homogeneous_scale moet groter dan 0 zijn."
            )

        self.calibration = calibration
        self.pitch_margin_m = float(
            pitch_margin_m
        )

        self.maximum_outside_distance_m = float(
            maximum_outside_distance_m
        )

        self.minimum_homogeneous_scale = float(
            minimum_homogeneous_scale
        )

        self._validate_calibration()

    @property
    def field_width_meters(self) -> float:
        """
        Breedte van het gekalibreerde speelveld.
        """

        return float(
            self.calibration.profile.width_m
        )

    @property
    def field_length_meters(self) -> float:
        """
        Lengte van het gekalibreerde speelveld.
        """

        return float(
            self.calibration.profile.length_m
        )

    @property
    def field_width(self) -> float:
        """
        Compatibiliteitsalias voor de veldbreedte.
        """

        return self.field_width_meters

    @property
    def field_length(self) -> float:
        """
        Compatibiliteitsalias voor de veldlengte.
        """

        return self.field_length_meters

    def project_point(
        self,
        image_point: tuple[float, float],
    ) -> FieldPosition | None:
        """
        Projecteer één beeldpunt naar een positie op het veld.

        Geeft None terug wanneer:

        - het beeldpunt NaN of Inf bevat;
        - de homogene projectieschaal vrijwel nul is;
        - OpenCV een ongeldige projectie produceert;
        - de veldpositie onrealistisch ver buiten het veld ligt.
        """

        validated_image_point = (
            self._validate_image_point(
                image_point=image_point,
            )
        )

        if validated_image_point is None:
            return None

        projected_point = (
            self._project_single_point(
                image_point=validated_image_point,
            )
        )

        if projected_point is None:
            return None

        x, y = projected_point

        return self._create_field_position(
            x=x,
            y=y,
        )

    def project_bounding_box(
        self,
        bounding_box: np.ndarray
        | tuple[float, float, float, float]
        | list[float],
    ) -> FieldPosition | None:
        """
        Projecteer het voetpunt van een bounding box.

        Bounding box-formaat:

            x1, y1, x2, y2

        Geeft None terug wanneer de bounding box of de resulterende
        veldprojectie ongeldig is.
        """

        box = np.asarray(
            bounding_box,
            dtype=np.float64,
        ).reshape(-1)

        if box.size != 4:
            raise ValueError(
                "Een bounding box moet exact vier waarden bevatten: "
                "x1, y1, x2, y2."
            )

        if not np.all(np.isfinite(box)):
            return None

        x1, y1, x2, y2 = (
            float(value)
            for value in box
        )

        if x2 < x1 or y2 < y1:
            return None

        if x2 == x1 or y2 == y1:
            return None

        foot_point = (
            (x1 + x2) / 2.0,
            y2,
        )

        return self.project_point(
            image_point=foot_point,
        )

    def project_points(
        self,
        image_points: Sequence[
            tuple[float, float]
        ],
    ) -> list[FieldPosition | None]:
        """
        Projecteer meerdere beeldpunten.

        De uitvoerlijst heeft exact dezelfde lengte en volgorde als de
        invoerlijst. Voor ongeldige punten wordt None opgenomen.
        """

        if not image_points:
            return []

        results: list[
            FieldPosition | None
        ] = []

        for image_point in image_points:
            results.append(
                self.project_point(
                    image_point=image_point,
                )
            )

        return results

    def is_inside_pitch(
        self,
        x: float,
        y: float,
    ) -> bool:
        """
        Controleer of een veldpositie binnen de veldgrenzen ligt.

        pitch_margin_m kan worden gebruikt om een kleine zone rond het
        officiële speelveld eveneens als binnen te behandelen.
        """

        if not isfinite(x) or not isfinite(y):
            return False

        width = self.field_width_meters
        length = self.field_length_meters
        margin = self.pitch_margin_m

        return (
            -margin <= x <= width + margin
            and -margin <= y <= length + margin
        )

    def distance_to_pitch(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Bereken de kortste afstand van een veldpositie tot het veld.

        Binnen de officiële veldrechthoek is de afstand 0 meter.
        pitch_margin_m wordt hier bewust niet meegenomen.
        """

        if not isfinite(x) or not isfinite(y):
            return float("inf")

        width = self.field_width_meters
        length = self.field_length_meters

        distance_x = max(
            0.0,
            -x,
            x - width,
        )

        distance_y = max(
            0.0,
            -y,
            y - length,
        )

        return float(
            np.hypot(
                distance_x,
                distance_y,
            )
        )

    def is_reasonable_field_position(
        self,
        x: float,
        y: float,
    ) -> bool:
        """
        Controleer of een geprojecteerde positie numeriek plausibel is.

        Posities binnen het veld zijn altijd plausibel. Posities buiten
        het veld zijn alleen plausibel wanneer de kortste afstand tot
        het veld niet groter is dan maximum_outside_distance_m.
        """

        if not isfinite(x) or not isfinite(y):
            return False

        outside_distance = (
            self.distance_to_pitch(
                x=x,
                y=y,
            )
        )

        return (
            outside_distance
            <= self.maximum_outside_distance_m
        )

    def distance_between(
        self,
        first: FieldPosition,
        second: FieldPosition,
    ) -> float:
        """
        Bereken de afstand in meters tussen twee veldposities.
        """

        return float(
            np.hypot(
                second.x - first.x,
                second.y - first.y,
            )
        )
    def _project_single_point(
        self,
        image_point: tuple[float, float],
    ) -> tuple[float, float] | None:
        """
        Voer de daadwerkelijke homografieprojectie uit.

        Vooraf wordt de homogene schaal gecontroleerd. Een schaal die
        vrijwel nul is, veroorzaakt extreem grote of instabiele
        projectieresultaten en wordt daarom afgewezen.
        """

        image_x, image_y = image_point

        homogeneous_point = np.asarray(
            [
                image_x,
                image_y,
                1.0,
            ],
            dtype=np.float64,
        )

        matrix = np.asarray(
            self.calibration.image_to_pitch_matrix,
            dtype=np.float64,
        )

        homogeneous_projection = (
            matrix @ homogeneous_point
        )

        if not np.all(
            np.isfinite(
                homogeneous_projection
            )
        ):
            return None

        homogeneous_scale = float(
            homogeneous_projection[2]
        )

        if (
            abs(homogeneous_scale)
            < self.minimum_homogeneous_scale
        ):
            return None

        image_array = np.asarray(
            [[image_point]],
            dtype=np.float32,
        )

        try:
            projected = cv2.perspectiveTransform(
                image_array,
                matrix,
            )[0, 0]
        except cv2.error:
            return None

        if projected.shape != (2,):
            return None

        if not np.all(
            np.isfinite(projected)
        ):
            return None

        x = float(projected[0])
        y = float(projected[1])

        if not isfinite(x) or not isfinite(y):
            return None

        return x, y

    def _create_field_position(
        self,
        x: float,
        y: float,
    ) -> FieldPosition | None:
        """
        Maak alleen een FieldPosition aan wanneer de projectie geldig en
        realistisch genoeg is.
        """

        if not self.is_reasonable_field_position(
            x=x,
            y=y,
        ):
            return None

        return FieldPosition(
            x=x,
            y=y,
            is_inside_pitch=self.is_inside_pitch(
                x=x,
                y=y,
            ),
        )

    @staticmethod
    def _validate_image_point(
        image_point: tuple[float, float],
    ) -> tuple[float, float] | None:
        """
        Valideer en normaliseer één beeldpunt.
        """

        try:
            point_array = np.asarray(
                image_point,
                dtype=np.float64,
            ).reshape(-1)
        except (
            TypeError,
            ValueError,
        ):
            return None

        if point_array.size != 2:
            return None

        if not np.all(
            np.isfinite(point_array)
        ):
            return None

        image_x = float(
            point_array[0]
        )

        image_y = float(
            point_array[1]
        )

        return image_x, image_y

    def _validate_calibration(self) -> None:
        """
        Controleer de kalibratiematrix en veldafmetingen.
        """

        matrix = np.asarray(
            self.calibration.image_to_pitch_matrix,
            dtype=np.float64,
        )

        if matrix.shape != (3, 3):
            raise ValueError(
                "image_to_pitch_matrix moet een 3x3-matrix zijn."
            )

        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "image_to_pitch_matrix bevat ongeldige waarden."
            )

        determinant = float(
            np.linalg.det(matrix)
        )

        if not isfinite(determinant):
            raise ValueError(
                "De determinant van image_to_pitch_matrix is ongeldig."
            )

        if abs(determinant) < 1e-12:
            raise ValueError(
                "image_to_pitch_matrix is niet omkeerbaar."
            )

        condition_number = float(
            np.linalg.cond(matrix)
        )

        if not isfinite(condition_number):
            raise ValueError(
                "image_to_pitch_matrix is numeriek instabiel."
            )

        if self.field_width_meters <= 0.0:
            raise ValueError(
                "De veldbreedte moet groter dan 0 zijn."
            )

        if self.field_length_meters <= 0.0:
            raise ValueError(
                "De veldlengte moet groter dan 0 zijn."
            )