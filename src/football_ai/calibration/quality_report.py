from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class ControlPointContext:
    """Herleidbare broninformatie van een kalibratiepunt."""

    landmark_key: int
    landmark_name: str
    frame_index: int
    frame_number: int


@dataclass(frozen=True)
class ErrorStatistics:
    """Samenvattende statistieken van een verzameling pixelafwijkingen."""

    point_count: int
    mean_error: float | None
    median_error: float | None
    rms_error: float | None
    max_error: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "point_count": self.point_count,
            "mean_error": self.mean_error,
            "median_error": self.median_error,
            "rms_error": self.rms_error,
            "max_error": self.max_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ErrorStatistics:
        return cls(
            point_count=int(data["point_count"]),
            mean_error=_optional_float(data.get("mean_error")),
            median_error=_optional_float(data.get("median_error")),
            rms_error=_optional_float(data.get("rms_error")),
            max_error=_optional_float(data.get("max_error")),
        )


@dataclass(frozen=True)
class PointReprojectionError:
    """Reprojectiefout van een individueel kalibratiepunt."""

    point_index: int
    observed_image_point: tuple[float, float]
    expected_pitch_point: tuple[float, float]
    reprojected_image_point: tuple[float, float]
    error_pixels: float
    is_inlier: bool
    landmark_key: int | None = None
    landmark_name: str | None = None
    frame_index: int | None = None
    frame_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_index": self.point_index,
            "observed_image_point": list(self.observed_image_point),
            "expected_pitch_point": list(self.expected_pitch_point),
            "reprojected_image_point": list(self.reprojected_image_point),
            "error_pixels": self.error_pixels,
            "is_inlier": self.is_inlier,
            "landmark_key": self.landmark_key,
            "landmark_name": self.landmark_name,
            "frame_index": self.frame_index,
            "frame_number": self.frame_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PointReprojectionError:
        return cls(
            point_index=int(data["point_index"]),
            observed_image_point=_point_from_json(
                data["observed_image_point"],
                "observed_image_point",
            ),
            expected_pitch_point=_point_from_json(
                data["expected_pitch_point"],
                "expected_pitch_point",
            ),
            reprojected_image_point=_point_from_json(
                data["reprojected_image_point"],
                "reprojected_image_point",
            ),
            error_pixels=float(data["error_pixels"]),
            is_inlier=bool(data["is_inlier"]),
            landmark_key=_optional_int(data.get("landmark_key")),
            landmark_name=data.get("landmark_name"),
            frame_index=_optional_int(data.get("frame_index")),
            frame_number=_optional_int(data.get("frame_number")),
        )


@dataclass(frozen=True)
class CalibrationQualityReport:
    """Kwaliteitsrapport voor alle controlepunten van een kalibratie."""

    point_errors: tuple[PointReprojectionError, ...]
    all_points: ErrorStatistics
    inlier_points: ErrorStatistics
    inlier_count: int
    outlier_count: int

    @property
    def point_count(self) -> int:
        return len(self.point_errors)

    def format_terminal_report(self) -> str:
        """Maak een compact, menselijk leesbaar terminalrapport."""
        lines = [
            "=" * 72,
            "Kalibratiekwaliteit - reprojectiefout in pixels",
            "=" * 72,
            f"Controlepunten : {self.point_count}",
            f"Inliers        : {self.inlier_count}",
            f"Outliers       : {self.outlier_count}",
            "-" * 72,
            (
                f"{'Statistiek':<16}"
                f"{'Alle punten':>18}"
                f"{'Alleen inliers':>20}"
            ),
            "-" * 72,
        ]

        metrics = (
            ("Gemiddelde", "mean_error"),
            ("Mediaan", "median_error"),
            ("RMS", "rms_error"),
            ("Maximum", "max_error"),
        )
        for label, attribute in metrics:
            lines.append(
                f"{label:<16}"
                f"{_format_pixels(getattr(self.all_points, attribute)):>18}"
                f"{_format_pixels(getattr(self.inlier_points, attribute)):>20}"
            )

        outliers = [
            point_error
            for point_error in self.point_errors
            if not point_error.is_inlier
        ]
        lines.extend(["-" * 72, "Outlierdetails"])
        if not outliers:
            lines.append("Geen outliers gedetecteerd.")
        else:
            lines.extend(
                _format_outlier(point_error)
                for point_error in outliers
            )

        lines.append("=" * 72)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_errors": [
                point_error.to_dict() for point_error in self.point_errors
            ],
            "all_points": self.all_points.to_dict(),
            "inlier_points": self.inlier_points.to_dict(),
            "inliers": self.inlier_count,
            "outliers": self.outlier_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationQualityReport:
        return cls(
            point_errors=tuple(
                PointReprojectionError.from_dict(point_error)
                for point_error in data["point_errors"]
            ),
            all_points=ErrorStatistics.from_dict(data["all_points"]),
            inlier_points=ErrorStatistics.from_dict(data["inlier_points"]),
            inlier_count=int(data["inliers"]),
            outlier_count=int(data["outliers"]),
        )


def calculate_quality_report(
    image_points: np.ndarray,
    pitch_points: np.ndarray,
    image_to_pitch_matrix: np.ndarray,
    inlier_mask: np.ndarray | None = None,
    point_contexts: Sequence[ControlPointContext] | None = None,
) -> CalibrationQualityReport:
    """
    Bereken pixel-reprojectiefouten voor een veldkalibratie.

    De homografie zet beeldpunten om naar meters. Voor een begrijpelijke en
    visueel controleerbare kwaliteitsmaat worden de bekende veldpunten met de
    inverse homografie teruggeprojecteerd naar het beeld. De Euclidische
    afstand tot het aangeklikte beeldpunt is de fout in pixels.

    Een ontbrekend inliermasker betekent dat alle punten als inlier gelden.
    """
    observed = _validate_points(image_points, "image_points")
    expected = _validate_points(pitch_points, "pitch_points")

    if len(observed) != len(expected):
        raise ValueError(
            "image_points en pitch_points moeten evenveel punten bevatten."
        )
    if len(observed) == 0:
        raise ValueError("Er is minimaal een controlepunt nodig.")

    matrix = _validate_homography(image_to_pitch_matrix)
    inliers = _normalise_inlier_mask(inlier_mask, len(observed))
    contexts = _normalise_point_contexts(point_contexts, len(observed))

    try:
        pitch_to_image = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("image_to_pitch_matrix is niet inverteerbaar.") from error

    reprojected = cv2.perspectiveTransform(
        expected.astype(np.float64).reshape(-1, 1, 2),
        pitch_to_image,
    ).reshape(-1, 2)

    if not np.all(np.isfinite(reprojected)):
        raise ValueError("De homografie produceert ongeldige reprojectiepunten.")

    errors = np.linalg.norm(observed - reprojected, axis=1)
    point_errors = tuple(
        PointReprojectionError(
            point_index=index,
            observed_image_point=_as_point(observed[index]),
            expected_pitch_point=_as_point(expected[index]),
            reprojected_image_point=_as_point(reprojected[index]),
            error_pixels=float(errors[index]),
            is_inlier=bool(inliers[index]),
            landmark_key=(
                contexts[index].landmark_key if contexts is not None else None
            ),
            landmark_name=(
                contexts[index].landmark_name if contexts is not None else None
            ),
            frame_index=(
                contexts[index].frame_index if contexts is not None else None
            ),
            frame_number=(
                contexts[index].frame_number if contexts is not None else None
            ),
        )
        for index in range(len(observed))
    )

    return CalibrationQualityReport(
        point_errors=point_errors,
        all_points=_calculate_statistics(errors),
        inlier_points=_calculate_statistics(errors[inliers]),
        inlier_count=int(np.count_nonzero(inliers)),
        outlier_count=int(np.count_nonzero(~inliers)),
    )


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    converted = np.asarray(points, dtype=np.float64)
    if converted.ndim != 2 or converted.shape[1:] != (2,):
        raise ValueError(f"{name} moet de vorm (n, 2) hebben.")
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} bevat ongeldige coördinaten.")
    return converted


def _validate_homography(matrix: np.ndarray) -> np.ndarray:
    converted = np.asarray(matrix, dtype=np.float64)
    if converted.shape != (3, 3):
        raise ValueError("image_to_pitch_matrix moet een 3x3-matrix zijn.")
    if not np.all(np.isfinite(converted)):
        raise ValueError("image_to_pitch_matrix bevat ongeldige waarden.")
    return converted


def _normalise_inlier_mask(
    inlier_mask: np.ndarray | None,
    point_count: int,
) -> np.ndarray:
    if inlier_mask is None:
        return np.ones(point_count, dtype=bool)

    converted = np.asarray(inlier_mask).reshape(-1)
    if len(converted) != point_count:
        raise ValueError(
            "inlier_mask moet exact een waarde per controlepunt bevatten."
        )
    return converted.astype(bool)


def _calculate_statistics(errors: np.ndarray) -> ErrorStatistics:
    values = np.asarray(errors, dtype=np.float64)
    if len(values) == 0:
        return ErrorStatistics(
            point_count=0,
            mean_error=None,
            median_error=None,
            rms_error=None,
            max_error=None,
        )

    return ErrorStatistics(
        point_count=len(values),
        mean_error=float(np.mean(values)),
        median_error=float(np.median(values)),
        rms_error=float(np.sqrt(np.mean(np.square(values)))),
        max_error=float(np.max(values)),
    )


def _as_point(point: np.ndarray) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def _point_from_json(value: Any, name: str) -> tuple[float, float]:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} moet twee geldige coördinaten bevatten.")
    return _as_point(point)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _format_pixels(value: float | None) -> str:
    return "n.v.t." if value is None else f"{value:.1f} px"


def _normalise_point_contexts(
    contexts: Sequence[ControlPointContext] | None,
    point_count: int,
) -> tuple[ControlPointContext, ...] | None:
    if contexts is None:
        return None
    converted = tuple(contexts)
    if len(converted) != point_count:
        raise ValueError(
            "point_contexts moet exact een context per controlepunt bevatten."
        )
    if not all(
        isinstance(context, ControlPointContext)
        for context in converted
    ):
        raise ValueError("point_contexts bevat een ongeldige context.")
    return converted


def _format_outlier(point_error: PointReprojectionError) -> str:
    parts = [f"Punt {point_error.point_index + 1}"]
    if point_error.landmark_name:
        parts.append(point_error.landmark_name)
    if point_error.frame_index is not None:
        parts.append(f"selectieframe {point_error.frame_index + 1}")
    if point_error.frame_number is not None:
        parts.append(f"videoframe {point_error.frame_number}")
    return " | ".join(parts) + f": {point_error.error_pixels:.1f} px"
