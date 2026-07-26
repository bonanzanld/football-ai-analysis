from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2, degrees

import cv2
import numpy as np

from football_ai.calibration.bootstrap.detection_profile import PitchDetectionProfile
from football_ai.calibration.bootstrap.white_line_detection import WhiteLineCandidate, detect_white_field_lines


class GroundLineFamily(str, Enum):
    LONGITUDINAL = "longitudinal"
    TRANSVERSE = "transverse"


@dataclass(frozen=True, slots=True)
class GroundLineEvidence:
    image_start: tuple[float, float]
    image_end: tuple[float, float]
    ground_start: tuple[float, float]
    ground_end: tuple[float, float]
    metric_length: float
    family: GroundLineFamily
    source_segments: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "image_start": list(self.image_start),
            "image_end": list(self.image_end),
            "ground_start": list(self.ground_start),
            "ground_end": list(self.ground_end),
            "metric_length": self.metric_length,
            "family": self.family.value,
            "source_segments": self.source_segments,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class GroundLineDetection:
    lines: tuple[GroundLineEvidence, ...]
    raw_candidates: int
    merged_candidates: int
    rejected_short: int
    rejected_direction: int
    rejected_implausible: int


def detect_metric_ground_lines(
    frame: np.ndarray,
    profile: PitchDetectionProfile,
    ground_to_image: np.ndarray,
    minimum_length_m: float = 3.0,
    direction_tolerance_degrees: float = 14.0,
) -> GroundLineDetection:
    """Merge white image fragments and retain long, pitch-aligned ground lines."""
    if minimum_length_m <= 0.0:
        raise ValueError("Minimale lijnlengte moet positief zijn.")
    homography = np.asarray(ground_to_image, dtype=np.float64)
    if homography.shape != (3, 3) or abs(float(np.linalg.det(homography))) < 1e-12:
        raise ValueError("ground_to_image moet een omkeerbare 3x3-homography zijn.")
    image_to_ground = np.linalg.inv(homography)
    visual = detect_white_field_lines(frame, profile)
    segments = [_metric_segment(item, image_to_ground) for item in visual.candidates]
    segments = [item for item in segments if item is not None]
    groups = _merge_collinear_segments(segments)
    accepted: list[GroundLineEvidence] = []
    rejected_short = 0
    rejected_direction = 0
    rejected_implausible = 0
    maximum_length = 2.0 * float(np.hypot(profile.pitch_length_m, profile.pitch_width_m))
    minimum_ground = np.asarray((-profile.pitch_length_m, -profile.pitch_width_m))
    maximum_ground = np.asarray((2.0 * profile.pitch_length_m, 2.0 * profile.pitch_width_m))
    for group in groups:
        points = np.vstack([item[0] for item in group] + [item[1] for item in group])
        start, end = _combined_endpoints(points)
        length = float(np.linalg.norm(end - start))
        if (
            length > maximum_length
            or np.any(start < minimum_ground)
            or np.any(start > maximum_ground)
            or np.any(end < minimum_ground)
            or np.any(end > maximum_ground)
        ):
            rejected_implausible += 1
            continue
        if length < minimum_length_m:
            rejected_short += 1
            continue
        family = _classify_direction(end - start, direction_tolerance_degrees)
        if family is None:
            rejected_direction += 1
            continue
        image_points = _project_points(np.vstack((start, end)), homography)
        confidence = float(np.mean([item[2].visual_confidence for item in group]))
        accepted.append(
            GroundLineEvidence(
                image_start=tuple(map(float, image_points[0])),
                image_end=tuple(map(float, image_points[1])),
                ground_start=tuple(map(float, start)),
                ground_end=tuple(map(float, end)),
                metric_length=length,
                family=family,
                source_segments=len(group),
                confidence=confidence,
            )
        )
    return GroundLineDetection(
        lines=tuple(sorted(accepted, key=lambda item: item.metric_length, reverse=True)),
        raw_candidates=len(visual.candidates),
        merged_candidates=len(groups),
        rejected_short=rejected_short,
        rejected_direction=rejected_direction,
        rejected_implausible=rejected_implausible,
    )


def draw_ground_line_evidence(frame: np.ndarray, detection: GroundLineDetection) -> np.ndarray:
    preview = frame.copy()
    colors = {
        GroundLineFamily.LONGITUDINAL: (255, 255, 0),
        GroundLineFamily.TRANSVERSE: (255, 0, 255),
    }
    for index, line in enumerate(detection.lines, start=1):
        start = tuple(np.round(line.image_start).astype(int))
        end = tuple(np.round(line.image_end).astype(int))
        color = colors[line.family]
        cv2.line(preview, start, end, color, 4, cv2.LINE_AA)
        midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        cv2.putText(
            preview,
            f"{index} {line.metric_length:.1f}m",
            midpoint,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return preview


def _metric_segment(
    candidate: WhiteLineCandidate,
    image_to_ground: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, WhiteLineCandidate] | None:
    points = _project_points(np.asarray((candidate.start, candidate.end), dtype=np.float64), image_to_ground)
    if not np.all(np.isfinite(points)) or np.linalg.norm(points[1] - points[0]) < 0.05:
        return None
    return points[0], points[1], candidate


def _merge_collinear_segments(
    segments: list[tuple[np.ndarray, np.ndarray, WhiteLineCandidate]],
) -> list[list[tuple[np.ndarray, np.ndarray, WhiteLineCandidate]]]:
    groups: list[list[tuple[np.ndarray, np.ndarray, WhiteLineCandidate]]] = []
    for segment in sorted(segments, key=lambda item: np.linalg.norm(item[1] - item[0]), reverse=True):
        for group in groups:
            if _same_metric_line(segment, group[0]):
                group.append(segment)
                break
        else:
            groups.append([segment])
    return groups


def _same_metric_line(
    first: tuple[np.ndarray, np.ndarray, WhiteLineCandidate],
    second: tuple[np.ndarray, np.ndarray, WhiteLineCandidate],
) -> bool:
    first_direction = first[1] - first[0]
    second_direction = second[1] - second[0]
    first_unit = first_direction / np.linalg.norm(first_direction)
    second_unit = second_direction / np.linalg.norm(second_direction)
    angle = degrees(np.arccos(np.clip(abs(float(first_unit @ second_unit)), 0.0, 1.0)))
    if angle > 6.0:
        return False
    normal = np.asarray((-first_unit[1], first_unit[0]))
    line_distance = min(
        abs(float((second[0] - first[0]) @ normal)),
        abs(float((second[1] - first[0]) @ normal)),
    )
    if line_distance > 0.45:
        return False
    axis_first = sorted((float(first[0] @ first_unit), float(first[1] @ first_unit)))
    axis_second = sorted((float(second[0] @ first_unit), float(second[1] @ first_unit)))
    gap = max(axis_first[0], axis_second[0]) - min(axis_first[1], axis_second[1])
    return gap <= 1.75


def _combined_endpoints(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    _u, _s, vh = np.linalg.svd(points - center)
    direction = vh[0]
    coordinates = (points - center) @ direction
    return center + direction * np.min(coordinates), center + direction * np.max(coordinates)


def _classify_direction(
    direction: np.ndarray,
    tolerance_degrees: float,
) -> GroundLineFamily | None:
    angle = abs(degrees(atan2(float(direction[1]), float(direction[0])))) % 180.0
    longitudinal_error = min(angle, 180.0 - angle)
    transverse_error = abs(angle - 90.0)
    if longitudinal_error <= tolerance_degrees:
        return GroundLineFamily.LONGITUDINAL
    if transverse_error <= tolerance_degrees:
        return GroundLineFamily.TRANSVERSE
    return None


def _project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (homography @ homogeneous.T).T
    valid = np.abs(projected[:, 2]) > 1e-12
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    result[valid] = projected[valid, :2] / projected[valid, 2:3]
    return result
