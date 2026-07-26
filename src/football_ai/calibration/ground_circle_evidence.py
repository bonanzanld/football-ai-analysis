from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from football_ai.calibration.bootstrap.detection_profile import PitchDetectionProfile
from football_ai.calibration.bootstrap.white_line_detection import extract_white_pitch_mask


@dataclass(frozen=True, slots=True)
class GroundCircleEvidence:
    ground_center: tuple[float, float]
    radius_m: float
    radial_support: float
    angular_coverage: float
    confidence: float
    halfway_line_support: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ground_center": list(self.ground_center),
            "radius_m": self.radius_m,
            "radial_support": self.radial_support,
            "angular_coverage": self.angular_coverage,
            "confidence": self.confidence,
            "halfway_line_support": self.halfway_line_support,
        }


@dataclass(frozen=True, slots=True)
class GroundCircleConsensus:
    ground_center: tuple[float, float]
    observations: int
    rms_m: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "ground_center": list(self.ground_center),
            "observations": self.observations,
            "rms_m": self.rms_m,
            "confidence": self.confidence,
        }


def estimate_circle_consensus(
    observations: tuple[GroundCircleEvidence, ...],
    cluster_radius_m: float = 2.0,
    minimum_observations: int = 3,
) -> GroundCircleConsensus | None:
    if cluster_radius_m <= 0.0 or minimum_observations < 1:
        raise ValueError("Clusterradius en minimaal aantal observaties moeten positief zijn.")
    if len(observations) < minimum_observations:
        return None
    centers = np.asarray([item.ground_center for item in observations], dtype=np.float64)
    best_indices: np.ndarray | None = None
    for center in centers:
        indices = np.flatnonzero(np.linalg.norm(centers - center, axis=1) <= cluster_radius_m)
        if best_indices is None or len(indices) > len(best_indices):
            best_indices = indices
    if best_indices is None or len(best_indices) < minimum_observations:
        return None
    selected = centers[best_indices]
    center = np.median(selected, axis=0)
    errors = np.linalg.norm(selected - center, axis=1)
    rms = float(np.sqrt(np.mean(np.square(errors))))
    confidences = [observations[index].confidence for index in best_indices]
    confidence = float(np.mean(confidences) * min(1.0, len(best_indices) / 5.0) * np.exp(-rms / 2.0))
    if rms > 1.5 or confidence < 0.35:
        return None
    return GroundCircleConsensus(tuple(map(float, center)), len(best_indices), rms, confidence)


def detect_metric_center_circle(
    frame: np.ndarray,
    profile: PitchDetectionProfile,
    ground_to_image: np.ndarray,
    expected_radius_m: float = 9.15,
    pixels_per_metre: float = 8.0,
) -> GroundCircleEvidence | None:
    """Detect the fixed 11v11 centre circle after rectifying the grass plane."""
    if expected_radius_m <= 0.0 or pixels_per_metre <= 0.0:
        raise ValueError("Cirkelstraal en rasterschaal moeten positief zijn.")
    homography = np.asarray(ground_to_image, dtype=np.float64)
    if homography.shape != (3, 3) or abs(float(np.linalg.det(homography))) < 1e-12:
        raise ValueError("ground_to_image moet een omkeerbare 3x3-homography zijn.")
    grass, white = extract_white_pitch_mask(frame)
    minimum = np.asarray((-profile.pitch_length_m, -profile.pitch_width_m), dtype=np.float64)
    maximum = np.asarray((2.0 * profile.pitch_length_m, 2.0 * profile.pitch_width_m), dtype=np.float64)
    canvas_size = np.ceil((maximum - minimum) * pixels_per_metre).astype(int)
    ground_to_canvas = np.asarray(
        (
            (pixels_per_metre, 0.0, -minimum[0] * pixels_per_metre),
            (0.0, pixels_per_metre, -minimum[1] * pixels_per_metre),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    image_to_canvas = ground_to_canvas @ np.linalg.inv(homography)
    rectified = cv2.warpPerspective(
        white,
        image_to_canvas,
        tuple(map(int, canvas_size)),
        flags=cv2.INTER_NEAREST,
    )
    evidence = detect_fixed_radius_circle_from_mask(
        rectified,
        expected_radius_m * pixels_per_metre,
    )
    if evidence is None:
        return None
    center_px, radial_support, angular_coverage = evidence
    halfway_support = _halfway_line_support(
        rectified,
        center_px,
        expected_radius_m * pixels_per_metre,
    )
    if halfway_support < 0.16:
        return None
    center_ground = np.asarray(center_px) / pixels_per_metre + minimum
    if not _circle_is_on_pitch_surface(
        center_ground,
        expected_radius_m,
        homography,
        grass,
        white,
    ):
        return None
    confidence = float(
        np.clip(0.45 * radial_support + 0.35 * angular_coverage + 0.20 * halfway_support, 0.0, 1.0)
    )
    if confidence < 0.42:
        return None
    return GroundCircleEvidence(
        tuple(map(float, center_ground)),
        expected_radius_m,
        radial_support,
        angular_coverage,
        confidence,
        halfway_support,
    )


def _halfway_line_support(
    rectified_white: np.ndarray,
    center_px: tuple[float, float],
    radius_px: float,
) -> float:
    x, y = center_px
    dilated = cv2.dilate(rectified_white, np.ones((7, 7), np.uint8))
    supports = []
    for direction in (-1.0, 1.0):
        distances = np.linspace(0.18 * radius_px, 1.35 * radius_px, 72)
        xs = np.round(np.full_like(distances, x)).astype(int)
        ys = np.round(y + direction * distances).astype(int)
        valid = (xs >= 0) & (xs < dilated.shape[1]) & (ys >= 0) & (ys < dilated.shape[0])
        supports.append(float(np.mean(dilated[ys[valid], xs[valid]] > 0)) if np.any(valid) else 0.0)
    return float(min(supports))


def validate_ground_circle_on_frame(
    evidence: GroundCircleEvidence,
    frame: np.ndarray,
    ground_to_image: np.ndarray,
) -> bool:
    grass, white = extract_white_pitch_mask(frame)
    return _circle_is_on_pitch_surface(
        np.asarray(evidence.ground_center, dtype=np.float64),
        evidence.radius_m,
        np.asarray(ground_to_image, dtype=np.float64),
        grass,
        white,
    )


def _circle_is_on_pitch_surface(
    center_ground: np.ndarray,
    radius_m: float,
    ground_to_image: np.ndarray,
    grass: np.ndarray,
    white: np.ndarray,
) -> bool:
    center = _project_ground_points(center_ground.reshape(1, 2), ground_to_image)[0]
    if not np.all(np.isfinite(center)):
        return False
    center_x, center_y = np.round(center).astype(int)
    if not (0 <= center_x < grass.shape[1] and 0 <= center_y < grass.shape[0]):
        return False
    radius = 9
    y0, y1 = max(0, center_y - radius), min(grass.shape[0], center_y + radius + 1)
    x0, x1 = max(0, center_x - radius), min(grass.shape[1], center_x + radius + 1)
    if float(np.mean(grass[y0:y1, x0:x1] > 0)) < 0.35:
        return False
    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    circumference = center_ground + radius_m * np.column_stack((np.cos(angles), np.sin(angles)))
    projected = _project_ground_points(circumference, ground_to_image)
    finite = np.all(np.isfinite(projected), axis=1)
    xs, ys = np.round(projected[:, 0]).astype(int), np.round(projected[:, 1]).astype(int)
    visible = finite & (xs >= 0) & (xs < grass.shape[1]) & (ys >= 0) & (ys < grass.shape[0])
    if float(np.mean(visible)) < 0.25:
        return False
    pitch_surface = cv2.dilate(cv2.bitwise_or(grass, white), np.ones((9, 9), np.uint8))
    if float(np.mean(pitch_surface[ys[visible], xs[visible]] > 0)) < 0.55:
        return False
    white_neighbourhood = cv2.dilate(white, np.ones((7, 7), np.uint8))
    return float(np.mean(white_neighbourhood[ys[visible], xs[visible]] > 0)) >= 0.12


def _project_ground_points(points: np.ndarray, ground_to_image: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (np.asarray(ground_to_image, dtype=np.float64) @ homogeneous.T).T
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.abs(projected[:, 2]) > 1e-12
    result[valid] = projected[valid, :2] / projected[valid, 2:3]
    return result


def detect_fixed_radius_circle_from_mask(
    mask: np.ndarray,
    radius_pixels: float,
) -> tuple[tuple[float, float], float, float] | None:
    if mask.ndim != 2 or radius_pixels <= 2.0:
        raise ValueError("Een enkelkanaalsmasker en een geldige cirkelstraal zijn vereist.")
    blurred = cv2.GaussianBlur(mask, (5, 5), 1.2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=max(2.0 * radius_pixels, 20.0),
        param1=80,
        param2=18,
        minRadius=max(3, round(radius_pixels * 0.88)),
        maxRadius=round(radius_pixels * 1.12),
    )
    if circles is None:
        return None
    best = None
    for x, y, radius in circles[0]:
        angles = np.linspace(0.0, 2.0 * np.pi, 144, endpoint=False)
        xs = np.round(x + radius * np.cos(angles)).astype(int)
        ys = np.round(y + radius * np.sin(angles)).astype(int)
        valid = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
        if np.count_nonzero(valid) < 36:
            continue
        supported = np.zeros(len(angles), dtype=bool)
        for dilation in (-3, -2, -1, 0, 1, 2, 3):
            sample_x = np.round(x + (radius + dilation) * np.cos(angles)).astype(int)
            sample_y = np.round(y + (radius + dilation) * np.sin(angles)).astype(int)
            inside = (sample_x >= 0) & (sample_x < mask.shape[1]) & (sample_y >= 0) & (sample_y < mask.shape[0])
            supported[inside] |= mask[sample_y[inside], sample_x[inside]] > 0
        radial_support = float(np.mean(supported[valid]))
        bins = supported.reshape(24, 6).any(axis=1)
        angular_coverage = float(np.mean(bins))
        score = 0.55 * radial_support + 0.45 * angular_coverage
        if best is None or score > best[0]:
            best = score, (float(x), float(y)), radial_support, angular_coverage
    return None if best is None else (best[1], best[2], best[3])


def project_ground_circle(
    evidence: GroundCircleEvidence,
    ground_to_image: np.ndarray,
    samples: int = 96,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    center = np.asarray(evidence.ground_center)
    ground = center + evidence.radius_m * np.column_stack((np.cos(angles), np.sin(angles)))
    homogeneous = np.column_stack((ground, np.ones(samples)))
    projected = (np.asarray(ground_to_image, dtype=np.float64) @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2:3]
