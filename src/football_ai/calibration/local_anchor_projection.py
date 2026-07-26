from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from football_ai.calibration.camera_projection_3d import CameraProjection3D
from football_ai.calibration.geometry_validation import ProjectedPitchGeometry, validate_projected_pitch_geometry
from football_ai.calibration.reference_3d import FootballFieldReference3D


@dataclass(frozen=True, slots=True)
class LocalProjectionResult:
    valid: bool
    projection: CameraProjection3D | None
    image_transform: np.ndarray | None
    good_matches: int
    inliers: int
    inlier_ratio: float
    anchor_coverage: float
    frame_coverage: float
    geometry: ProjectedPitchGeometry | None
    reason: str


def estimate_local_anchor_projection(
    anchor_frame: np.ndarray,
    frame: np.ndarray,
    anchor_projection: CameraProjection3D,
    reference: FootballFieldReference3D,
) -> LocalProjectionResult:
    """Move one trusted projection to a visually nearby frame.

    This function never chains transforms through prior video frames. Every
    estimate is calculated directly from the immutable anchor image.
    """
    orb = cv2.ORB_create(nfeatures=4500, fastThreshold=12)
    anchor_keypoints, anchor_descriptors = orb.detectAndCompute(
        cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY), None
    )
    frame_keypoints, frame_descriptors = orb.detectAndCompute(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None
    )
    if anchor_descriptors is None or frame_descriptors is None:
        return _failure("Te weinig beeldkenmerken voor lokale koppeling.")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(anchor_descriptors, frame_descriptors, k=2)
    good = [first for pair in pairs if len(pair) == 2 for first, second in [pair] if first.distance < 0.72 * second.distance]
    if len(good) < 30:
        return _failure("Minder dan 30 betrouwbare lokale overeenkomsten.", good_matches=len(good))
    source = np.float32([anchor_keypoints[item.queryIdx].pt for item in good])
    target = np.float32([frame_keypoints[item.trainIdx].pt for item in good])
    transform, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.5)
    if transform is None or mask is None or not np.all(np.isfinite(transform)):
        return _failure("Lokale beeldtransformatie kon niet worden bepaald.", good_matches=len(good))
    selected = mask.ravel().astype(bool)
    inliers = int(np.count_nonzero(selected))
    ratio = inliers / len(good)
    anchor_coverage = _coverage(source[selected], anchor_frame.shape[1], anchor_frame.shape[0])
    frame_coverage = _coverage(target[selected], frame.shape[1], frame.shape[0])
    statistics = dict(
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=float(ratio),
        anchor_coverage=anchor_coverage,
        frame_coverage=frame_coverage,
    )
    if inliers < 25 or ratio < 0.40:
        return _failure("Lokale koppeling heeft te weinig geometrische inliers.", **statistics)
    if anchor_coverage < 0.06 or frame_coverage < 0.06:
        return _failure("Lokale inliers liggen te geconcentreerd in het beeld.", **statistics)
    transform /= transform[2, 2]
    if not _transform_is_sane(transform, anchor_frame.shape[1], anchor_frame.shape[0]):
        return _failure("Lokale beeldtransformatie vervormt het anker onrealistisch.", **statistics)
    projection = CameraProjection3D(transform @ anchor_projection.matrix)
    field_ids = ("corner_a_rear", "corner_b_rear", "corner_b_front", "corner_a_front")
    try:
        corners = np.asarray(
            [projection.project(reference.landmark(item).point) for item in field_ids],
            dtype=np.float64,
        )
    except ValueError as error:
        return _failure(str(error), **statistics)
    geometry = validate_projected_pitch_geometry(corners, frame.shape[1], frame.shape[0])
    if not geometry.valid:
        return LocalProjectionResult(
            False, None, transform, geometry=geometry,
            reason=" ".join(geometry.errors), **statistics,
        )
    return LocalProjectionResult(
        True, projection, transform, geometry=geometry,
        reason="Lokale projectie is geometrisch geldig.", **statistics,
    )


def _coverage(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3:
        return 0.0
    return abs(float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))) / max(float(width * height), 1.0)


def _transform_is_sane(transform: np.ndarray, width: int, height: int) -> bool:
    corners = np.float32(((0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0)))
    warped = cv2.perspectiveTransform(corners.reshape(1, -1, 2), transform).reshape(-1, 2)
    if not np.all(np.isfinite(warped)) or not cv2.isContourConvex(warped.astype(np.float32).reshape(-1, 1, 2)):
        return False
    ratio = abs(float(cv2.contourArea(warped.astype(np.float32)))) / max(float(width * height), 1.0)
    return 0.35 <= ratio <= 2.85


def _failure(
    reason: str,
    good_matches: int = 0,
    inliers: int = 0,
    inlier_ratio: float = 0.0,
    anchor_coverage: float = 0.0,
    frame_coverage: float = 0.0,
) -> LocalProjectionResult:
    return LocalProjectionResult(
        False, None, None, good_matches, inliers, inlier_ratio,
        anchor_coverage, frame_coverage, None, reason,
    )
