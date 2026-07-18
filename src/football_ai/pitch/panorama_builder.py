"""
panorama_builder.py

Analyseert meerdere videoframes en hun bijbehorende transformaties voor het
bouwen van een panorama.

Sprint 1B bevat:

- validatie van 3x3-transformaties;
- een 11x11 raster met samplepunten per frame;
- projectie van alle samplepunten;
- detectie van extreme projectieve vervorming;
- berekening van een bounding box per frame;
- berekening van de verwachte panoramagrootte.

Deze module maakt nog geen panorama-afbeelding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class PanoramaFrameReport:
    """
    Geometrisch rapport van één frame-transformatie.
    """

    frame_index: int
    valid: bool = True
    affine_fallback_recommended: bool = False

    image_width: int = 0
    image_height: int = 0

    determinant: float = 0.0
    perspective_strength: float = 0.0

    sample_count: int = 0
    projected_sample_count: int = 0

    minimum_x: float = 0.0
    minimum_y: float = 0.0
    maximum_x: float = 0.0
    maximum_y: float = 0.0

    projected_width: float = 0.0
    projected_height: float = 0.0

    warnings: list[str] = field(default_factory=list)

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """
        Bounding box als:

        minimum_x, minimum_y, maximum_x, maximum_y
        """
        return (
            self.minimum_x,
            self.minimum_y,
            self.maximum_x,
            self.maximum_y,
        )


@dataclass(slots=True)
class PanoramaReport:
    """
    Geometrisch rapport van alle frames samen.
    """

    overall_valid: bool = True

    canvas_minimum_x: float = 0.0
    canvas_minimum_y: float = 0.0
    canvas_maximum_x: float = 0.0
    canvas_maximum_y: float = 0.0

    canvas_width: int = 0
    canvas_height: int = 0

    frame_reports: list[PanoramaFrameReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_frame_report(
        self,
        frame_report: PanoramaFrameReport,
    ) -> None:
        self.frame_reports.append(frame_report)

        if not frame_report.valid:
            self.overall_valid = False

    def print_summary(self) -> None:
        """
        Print een leesbaar analyserapport in de terminal.
        """
        print("=" * 78)
        print("PanoramaBuilder - Sprint 1B geometrisch rapport")
        print("=" * 78)

        print(f"Aantal frames : {len(self.frame_reports)}")
        print(
            "Totale status : "
            + ("GELDIG" if self.overall_valid else "ONGELDIG")
        )
        print(
            f"Canvas        : {self.canvas_width} x "
            f"{self.canvas_height} pixels"
        )
        print(
            "Canvasgebied  : "
            f"x={self.canvas_minimum_x:.1f} tot "
            f"{self.canvas_maximum_x:.1f}, "
            f"y={self.canvas_minimum_y:.1f} tot "
            f"{self.canvas_maximum_y:.1f}"
        )

        for warning in self.warnings:
            print(f"WAARSCHUWING : {warning}")

        print("-" * 78)

        for frame_report in self.frame_reports:
            status = "OK" if frame_report.valid else "FOUT"

            if frame_report.affine_fallback_recommended:
                status += " / AFFINE FALLBACK AANBEVOLEN"

            print(
                f"Frame {frame_report.frame_index + 1}: {status}"
            )
            print(
                f"  Origineel       : "
                f"{frame_report.image_width} x "
                f"{frame_report.image_height}"
            )
            print(
                f"  Projectiegebied : "
                f"{frame_report.projected_width:.1f} x "
                f"{frame_report.projected_height:.1f}"
            )
            print(
                f"  Bounding box    : "
                f"({frame_report.minimum_x:.1f}, "
                f"{frame_report.minimum_y:.1f}) - "
                f"({frame_report.maximum_x:.1f}, "
                f"{frame_report.maximum_y:.1f})"
            )
            print(
                f"  Samples         : "
                f"{frame_report.projected_sample_count}/"
                f"{frame_report.sample_count}"
            )
            print(
                f"  Determinant     : "
                f"{frame_report.determinant:.6e}"
            )
            print(
                f"  Perspectief     : "
                f"{frame_report.perspective_strength:.6e}"
            )

            for warning in frame_report.warnings:
                print(f"  - {warning}")

            print("-" * 78)

        print("=" * 78)


class PanoramaBuilder:
    """
    Analyseert frames en hun transformaties.

    De klasse verwacht:

    - selected_frames:
      een lijst met objecten die een attribuut `.frame` bevatten;

    - frame_transforms:
      een lijst met 3x3 NumPy-matrices die ieder frame naar het
      referentieframe transformeren.

    Sprint 1B berekent alleen de geometrie. Er wordt nog geen panorama gebouwd.
    """

    SAMPLE_GRID_SIZE = 11

    MINIMUM_DETERMINANT = 1e-10
    MAXIMUM_PERSPECTIVE_STRENGTH = 0.01

    MAXIMUM_PROJECTED_SCALE = 8.0
    MAXIMUM_CANVAS_DIMENSION = 10_000

    CANVAS_PADDING = 80

    def __init__(
        self,
        selected_frames: list[Any],
        frame_transforms: list[np.ndarray],
    ) -> None:
        self.selected_frames = selected_frames
        self.frame_transforms = frame_transforms

    def analyze(self) -> PanoramaReport:
        """
        Analyseer alle frames en bereken het verwachte panoramocanvas.

        Returns:
            PanoramaReport met één rapport per frame en de gezamenlijke
            canvasafmetingen.

        Raises:
            ValueError wanneer de invoerstructuur niet geldig is.
        """
        self._validate_input()

        report = PanoramaReport()

        for frame_index, (selected_frame, transform) in enumerate(
            zip(self.selected_frames, self.frame_transforms)
        ):
            image = selected_frame.frame

            frame_report = self._analyze_frame_transform(
                image_shape=image.shape,
                transform=transform,
                frame_index=frame_index,
            )

            report.add_frame_report(frame_report)

        self._compute_canvas(report)
        return report

    def _validate_input(self) -> None:
        """
        Controleer de algemene invoer.
        """
        if not self.selected_frames:
            raise ValueError(
                "Er zijn geen geselecteerde frames aangeleverd."
            )

        if len(self.selected_frames) != len(self.frame_transforms):
            raise ValueError(
                "Het aantal geselecteerde frames komt niet overeen met "
                "het aantal frame-transformaties."
            )

        for frame_index, selected_frame in enumerate(
            self.selected_frames
        ):
            image = getattr(selected_frame, "frame", None)

            if image is None:
                raise ValueError(
                    f"Frame {frame_index + 1} bevat geen `.frame` afbeelding."
                )

            if not isinstance(image, np.ndarray):
                raise ValueError(
                    f"Frame {frame_index + 1} is geen NumPy-array."
                )

            if image.ndim not in (2, 3):
                raise ValueError(
                    f"Frame {frame_index + 1} heeft een ongeldige vorm: "
                    f"{image.shape}."
                )

            if image.shape[0] <= 0 or image.shape[1] <= 0:
                raise ValueError(
                    f"Frame {frame_index + 1} heeft ongeldige afmetingen."
                )

    def _analyze_frame_transform(
        self,
        image_shape: tuple[int, ...],
        transform: np.ndarray,
        frame_index: int,
    ) -> PanoramaFrameReport:
        """
        Analyseer één frame-transformatie.
        """
        image_height = int(image_shape[0])
        image_width = int(image_shape[1])

        frame_report = PanoramaFrameReport(
            frame_index=frame_index,
            image_width=image_width,
            image_height=image_height,
        )

        matrix = self._validate_transform(
            transform=transform,
            frame_report=frame_report,
        )

        if matrix is None:
            return frame_report

        sample_points = self._create_sample_grid(
            image_width=image_width,
            image_height=image_height,
        )

        frame_report.sample_count = len(sample_points)

        projected_points = self._project_sample_points(
            sample_points=sample_points,
            transform=matrix,
            frame_report=frame_report,
        )

        frame_report.projected_sample_count = len(projected_points)

        if len(projected_points) < 4:
            frame_report.valid = False
            frame_report.warnings.append(
                "Te weinig geldige projectiepunten om een bounding box "
                "te berekenen."
            )
            return frame_report

        self._set_projected_bounding_box(
            projected_points=projected_points,
            frame_report=frame_report,
        )

        self._evaluate_projected_geometry(frame_report)
        return frame_report

    def _validate_transform(
        self,
        transform: np.ndarray,
        frame_report: PanoramaFrameReport,
    ) -> np.ndarray | None:
        """
        Controleer en normaliseer één 3x3-transformatie.
        """
        if not isinstance(transform, np.ndarray):
            frame_report.valid = False
            frame_report.warnings.append(
                "De transformatie is geen NumPy-array."
            )
            return None

        if transform.shape != (3, 3):
            frame_report.valid = False
            frame_report.warnings.append(
                f"De transformatie heeft vorm {transform.shape}, "
                "maar moet 3x3 zijn."
            )
            return None

        if not np.all(np.isfinite(transform)):
            frame_report.valid = False
            frame_report.warnings.append(
                "De transformatie bevat NaN of oneindige waarden."
            )
            return None

        matrix = transform.astype(np.float64)

        determinant = float(np.linalg.det(matrix))
        frame_report.determinant = determinant

        if abs(determinant) < self.MINIMUM_DETERMINANT:
            frame_report.valid = False
            frame_report.warnings.append(
                "De transformatie is singulier of bijna singulier."
            )
            return None

        if abs(matrix[2, 2]) < 1e-12:
            frame_report.valid = False
            frame_report.warnings.append(
                "De schaalwaarde rechtsonder in de matrix is vrijwel nul."
            )
            return None

        matrix /= matrix[2, 2]

        perspective_strength = float(
            np.linalg.norm(matrix[2, :2])
        )
        frame_report.perspective_strength = perspective_strength

        if (
            perspective_strength
            > self.MAXIMUM_PERSPECTIVE_STRENGTH
        ):
            frame_report.warnings.append(
                "De transformatie bevat relatief sterke projectieve "
                "vervorming."
            )

        return matrix

    def _create_sample_grid(
        self,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """
        Maak een regelmatig raster van samplepunten over het hele frame.

        Bij SAMPLE_GRID_SIZE = 11 worden 121 punten aangemaakt.
        """
        x_values = np.linspace(
            0.0,
            float(image_width - 1),
            self.SAMPLE_GRID_SIZE,
            dtype=np.float32,
        )

        y_values = np.linspace(
            0.0,
            float(image_height - 1),
            self.SAMPLE_GRID_SIZE,
            dtype=np.float32,
        )

        grid_x, grid_y = np.meshgrid(x_values, y_values)

        return np.column_stack(
            (
                grid_x.reshape(-1),
                grid_y.reshape(-1),
            )
        ).astype(np.float32)

    @staticmethod
    def _project_sample_points(
        sample_points: np.ndarray,
        transform: np.ndarray,
        frame_report: PanoramaFrameReport,
    ) -> np.ndarray:
        """
        Projecteer de samplepunten met de frame-transformatie.

        Ongeldige punten met NaN of Inf worden verwijderd.
        """
        try:
            projected = cv2.perspectiveTransform(
                sample_points.reshape(-1, 1, 2),
                transform,
            ).reshape(-1, 2)
        except cv2.error as error:
            frame_report.valid = False
            frame_report.warnings.append(
                f"OpenCV kon de samplepunten niet projecteren: {error}"
            )
            return np.empty((0, 2), dtype=np.float32)

        finite_mask = np.all(np.isfinite(projected), axis=1)
        valid_points = projected[finite_mask]

        removed_count = len(projected) - len(valid_points)

        if removed_count > 0:
            frame_report.warnings.append(
                f"{removed_count} samplepunten leverden NaN of Inf op."
            )

        return valid_points

    @staticmethod
    def _set_projected_bounding_box(
        projected_points: np.ndarray,
        frame_report: PanoramaFrameReport,
    ) -> None:
        """
        Bereken de bounding box van de geprojecteerde samplepunten.
        """
        minimum = np.min(projected_points, axis=0)
        maximum = np.max(projected_points, axis=0)

        frame_report.minimum_x = float(minimum[0])
        frame_report.minimum_y = float(minimum[1])
        frame_report.maximum_x = float(maximum[0])
        frame_report.maximum_y = float(maximum[1])

        frame_report.projected_width = float(
            maximum[0] - minimum[0]
        )
        frame_report.projected_height = float(
            maximum[1] - minimum[1]
        )

    def _evaluate_projected_geometry(
        self,
        frame_report: PanoramaFrameReport,
    ) -> None:
        """
        Bepaal of het geprojecteerde frame onrealistisch groot wordt.
        """
        original_width = max(frame_report.image_width, 1)
        original_height = max(frame_report.image_height, 1)

        width_scale = (
            frame_report.projected_width / original_width
        )
        height_scale = (
            frame_report.projected_height / original_height
        )

        if frame_report.projected_width <= 0:
            frame_report.valid = False
            frame_report.warnings.append(
                "Het geprojecteerde frame heeft geen geldige breedte."
            )

        if frame_report.projected_height <= 0:
            frame_report.valid = False
            frame_report.warnings.append(
                "Het geprojecteerde frame heeft geen geldige hoogte."
            )

        if (
            width_scale > self.MAXIMUM_PROJECTED_SCALE
            or height_scale > self.MAXIMUM_PROJECTED_SCALE
        ):
            frame_report.affine_fallback_recommended = True
            frame_report.warnings.append(
                "Het geprojecteerde beeldgebied is extreem groot. "
                "Een affine fallback wordt aanbevolen voor het panorama."
            )

        maximum_coordinate = max(
            abs(frame_report.minimum_x),
            abs(frame_report.minimum_y),
            abs(frame_report.maximum_x),
            abs(frame_report.maximum_y),
        )

        reference_size = max(
            frame_report.image_width,
            frame_report.image_height,
        )

        if maximum_coordinate > reference_size * 20:
            frame_report.affine_fallback_recommended = True
            frame_report.warnings.append(
                "De projectie bevat extreme coördinaten buiten het "
                "referentiebeeld."
            )

    def _compute_canvas(
        self,
        report: PanoramaReport,
    ) -> None:
        """
        Bereken het gezamenlijke panoramocanvas uit alle geldige frames.
        """
        usable_reports = [
            frame_report
            for frame_report in report.frame_reports
            if (
                frame_report.valid
                and frame_report.projected_width > 0
                and frame_report.projected_height > 0
            )
        ]

        if not usable_reports:
            report.overall_valid = False
            report.warnings.append(
                "Geen geldige frames beschikbaar voor canvasberekening."
            )
            return

        minimum_x = min(
            frame_report.minimum_x
            for frame_report in usable_reports
        )
        minimum_y = min(
            frame_report.minimum_y
            for frame_report in usable_reports
        )
        maximum_x = max(
            frame_report.maximum_x
            for frame_report in usable_reports
        )
        maximum_y = max(
            frame_report.maximum_y
            for frame_report in usable_reports
        )

        report.canvas_minimum_x = minimum_x
        report.canvas_minimum_y = minimum_y
        report.canvas_maximum_x = maximum_x
        report.canvas_maximum_y = maximum_y

        report.canvas_width = int(
            np.ceil(maximum_x - minimum_x)
            + 2 * self.CANVAS_PADDING
        )

        report.canvas_height = int(
            np.ceil(maximum_y - minimum_y)
            + 2 * self.CANVAS_PADDING
        )

        if (
            report.canvas_width > self.MAXIMUM_CANVAS_DIMENSION
            or report.canvas_height > self.MAXIMUM_CANVAS_DIMENSION
        ):
            report.warnings.append(
                "Het verwachte panoramocanvas is groter dan "
                f"{self.MAXIMUM_CANVAS_DIMENSION} pixels. "
                "Minimaal één frame heeft waarschijnlijk een instabiele "
                "projectieve transformatie."
            )

        if report.canvas_width <= 0 or report.canvas_height <= 0:
            report.overall_valid = False
            report.warnings.append(
                "De berekende canvasafmetingen zijn ongeldig."
            )