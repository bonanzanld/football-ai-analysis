from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees

import cv2
import numpy as np

from football_ai.calibration.bootstrap.detection_profile import PitchDetectionProfile


@dataclass(frozen=True, slots=True)
class WhiteLineCandidate:
    start: tuple[float, float]
    end: tuple[float, float]
    angle_degrees: float
    length_pixels: float
    white_support: float
    grass_context: float
    visual_confidence: float
    profile_evidence: float

    def to_dict(self) -> dict:
        return {
            "start": list(self.start),
            "end": list(self.end),
            "angle_degrees": self.angle_degrees,
            "length_pixels": self.length_pixels,
            "white_support": self.white_support,
            "grass_context": self.grass_context,
            "visual_confidence": self.visual_confidence,
            "profile_evidence": self.profile_evidence,
        }


@dataclass(frozen=True, slots=True)
class WhiteLineDetection:
    candidates: tuple[WhiteLineCandidate, ...]
    grass_coverage: float
    white_pixel_ratio: float


def detect_white_field_lines(
    frame: np.ndarray,
    profile: PitchDetectionProfile,
) -> WhiteLineDetection:
    """Vind rechte witte lijnkandidaten met grascontext, zonder camera-aannames."""
    grass, white_on_pitch = extract_white_pitch_mask(frame)
    edges = cv2.Canny(white_on_pitch, 45, 135)
    minimum_length = max(28, int(min(frame.shape[:2]) * 0.055))
    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720.0,
        threshold=24,
        minLineLength=minimum_length,
        maxLineGap=24,
    )
    candidates: list[WhiteLineCandidate] = []
    for packed in ([] if raw is None else raw):
        x1, y1, x2, y2 = map(float, packed.reshape(4))
        length = float(np.hypot(x2 - x1, y2 - y1))
        white_support = _line_mask_support(white_on_pitch, (x1, y1), (x2, y2), 4)
        grass_context = _parallel_grass_support(grass, (x1, y1), (x2, y2))
        if white_support < 0.58 or grass_context < 0.28:
            continue
        length_score = min(1.0, length / (frame.shape[1] * 0.28))
        visual = float(
            np.clip(0.48 * white_support + 0.30 * grass_context + 0.22 * length_score, 0.0, 1.0)
        )
        candidates.append(
            WhiteLineCandidate(
                start=(x1, y1),
                end=(x2, y2),
                angle_degrees=degrees(atan2(y2 - y1, x2 - x1)),
                length_pixels=length,
                white_support=white_support,
                grass_context=grass_context,
                visual_confidence=visual,
                profile_evidence=visual * profile.white_line_evidence_weight,
            )
        )
    candidates = _suppress_duplicate_segments(candidates)
    return WhiteLineDetection(
        candidates=tuple(sorted(candidates, key=lambda item: item.visual_confidence, reverse=True)),
        grass_coverage=float(np.count_nonzero(grass) / grass.size),
        white_pixel_ratio=float(np.count_nonzero(white_on_pitch) / white_on_pitch.size),
    )


def extract_white_pitch_mask(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the largest grass component and bright markings touching that surface."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (28, 32, 25), (86, 255, 235))
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    grass = _largest_component(grass)
    value_floor = max(155, int(np.percentile(hsv[..., 2], 82)))
    white = cv2.inRange(hsv, (0, 0, value_floor), (180, 62, 255))
    pitch_neighbourhood = cv2.dilate(grass, np.ones((17, 17), np.uint8))
    white_on_pitch = cv2.bitwise_and(white, pitch_neighbourhood)
    white_on_pitch = cv2.morphologyEx(
        white_on_pitch,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    return grass, white_on_pitch


def draw_white_line_detection(
    frame: np.ndarray,
    detection: WhiteLineDetection,
) -> np.ndarray:
    overlay = frame.copy()
    for index, candidate in enumerate(detection.candidates, start=1):
        color = (40, 220, 40) if candidate.visual_confidence >= 0.65 else (0, 180, 255)
        start = tuple(np.round(candidate.start).astype(int))
        end = tuple(np.round(candidate.end).astype(int))
        cv2.line(overlay, start, end, color, 3, cv2.LINE_AA)
        midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        cv2.putText(
            overlay,
            f"{index}:{candidate.visual_confidence:.2f}",
            midpoint,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def _line_mask_support(
    mask: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    thickness: int,
) -> float:
    probe = np.zeros(mask.shape, dtype=np.uint8)
    cv2.line(probe, tuple(map(round, start)), tuple(map(round, end)), 255, thickness)
    selected = probe > 0
    return float(np.mean(mask[selected] > 0)) if np.any(selected) else 0.0


def _parallel_grass_support(
    grass: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    direction = np.asarray(end) - np.asarray(start)
    length = float(np.linalg.norm(direction))
    if length < 1.0:
        return 0.0
    normal = np.asarray([-direction[1], direction[0]]) / length
    supports = []
    for offset in (-8.0, 8.0):
        shifted_start = np.asarray(start) + normal * offset
        shifted_end = np.asarray(end) + normal * offset
        supports.append(_line_mask_support(grass, tuple(shifted_start), tuple(shifted_end), 3))
    return float(np.mean(supports))


def _suppress_duplicate_segments(
    candidates: list[WhiteLineCandidate],
) -> list[WhiteLineCandidate]:
    kept: list[WhiteLineCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.visual_confidence, reverse=True):
        midpoint = (np.asarray(candidate.start) + np.asarray(candidate.end)) / 2.0
        duplicate = False
        for other in kept:
            other_midpoint = (np.asarray(other.start) + np.asarray(other.end)) / 2.0
            angle_difference = abs(candidate.angle_degrees - other.angle_degrees) % 180.0
            angle_difference = min(angle_difference, 180.0 - angle_difference)
            if angle_difference < 4.0 and np.linalg.norm(midpoint - other_midpoint) < 18.0:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _largest_component(mask: np.ndarray) -> np.ndarray:
    component_count, labels, statistics, _centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    if component_count <= 1:
        return mask
    largest_label = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    result = np.zeros_like(mask)
    result[labels == largest_label] = 255
    return result
