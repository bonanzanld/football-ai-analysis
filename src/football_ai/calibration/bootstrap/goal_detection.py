from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class GoalCandidate:
    left_ground: tuple[float, float]
    right_ground: tuple[float, float]
    left_top: tuple[float, float]
    right_top: tuple[float, float]
    confidence: float
    crossbar_supported: bool
    backline_support: float

    @property
    def center_ground(self) -> tuple[float, float]:
        return (
            (self.left_ground[0] + self.right_ground[0]) / 2.0,
            (self.left_ground[1] + self.right_ground[1]) / 2.0,
        )

    def to_dict(self) -> dict:
        return {
            "left_ground": list(self.left_ground),
            "right_ground": list(self.right_ground),
            "left_top": list(self.left_top),
            "right_top": list(self.right_top),
            "center_ground": list(self.center_ground),
            "confidence": self.confidence,
            "crossbar_supported": self.crossbar_supported,
            "backline_support": self.backline_support,
        }


@dataclass(frozen=True, slots=True)
class GoalDetection:
    candidates: tuple[GoalCandidate, ...]


def measure_backline_support(
    frame: np.ndarray,
    first_ground: tuple[float, float],
    second_ground: tuple[float, float],
) -> float:
    """Meet witte-lijnondersteuning door twee handmatig bevestigde doelpalen."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value_floor = max(145, int(np.percentile(hsv[..., 2], 78)))
    white = cv2.inRange(hsv, (0, 0, value_floor), (180, 72, 255))
    return _backline_support(
        white,
        np.asarray(first_ground, dtype=np.float64),
        np.asarray(second_ground, dtype=np.float64),
    )


def detect_goal_candidates(frame: np.ndarray) -> GoalDetection:
    """Zoek witte paalparen met vergelijkbare grondhoogte en een dwarslat."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value_floor = max(145, int(np.percentile(hsv[..., 2], 78)))
    white = cv2.inRange(hsv, (0, 0, value_floor), (180, 72, 255))
    grass = cv2.inRange(hsv, (28, 32, 25), (86, 255, 235))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    edges = cv2.Canny(white, 45, 140)
    minimum_post_length = max(12, int(frame.shape[0] * 0.018))
    packed = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720.0,
        threshold=14,
        minLineLength=minimum_post_length,
        maxLineGap=10,
    )
    segments = [] if packed is None else [item.reshape(4).astype(float) for item in packed]
    vertical = [item for item in segments if _vertical_deviation(item) <= 18.0]
    horizontal = [item for item in segments if _horizontal_deviation(item) <= 14.0]
    posts = [_ordered_post(item) for item in vertical]
    results: list[GoalCandidate] = []
    minimum_width = max(18.0, frame.shape[1] * 0.018)
    maximum_width = frame.shape[1] * 0.55
    for left_index, first in enumerate(posts[:-1]):
        for second in posts[left_index + 1:]:
            left, right = (first, second) if first[0][0] <= second[0][0] else (second, first)
            width = right[0][0] - left[0][0]
            if not minimum_width <= width <= maximum_width:
                continue
            mean_height = (left[2] + right[2]) / 2.0
            if mean_height < minimum_post_length:
                continue
            mean_ground_y = (left[0][1] + right[0][1]) / 2.0
            if mean_ground_y > frame.shape[0] * 0.52:
                continue
            if abs(left[0][1] - right[0][1]) > max(18.0, mean_height * 0.55):
                continue
            if abs(left[1][1] - right[1][1]) > max(18.0, mean_height * 0.55):
                continue
            crossbar = _crossbar_support(horizontal, left[1], right[1])
            height_similarity = 1.0 - min(1.0, abs(left[2] - right[2]) / max(mean_height, 1.0))
            ground_similarity = 1.0 - min(1.0, abs(left[0][1] - right[0][1]) / max(mean_height, 1.0))
            aspect_ratio = width / max(mean_height, 1.0)
            if not 0.9 <= aspect_ratio <= 4.8:
                continue
            if not _goal_ground_has_grass(grass, left[0], right[0]):
                continue
            backline_support = _backline_support(white, left[0], right[0])
            if backline_support < 0.24:
                continue
            aspect_score = float(np.exp(-abs(aspect_ratio - 2.2) / 2.2))
            confidence = float(
                np.clip(
                    0.28 * height_similarity
                    + 0.24 * ground_similarity
                    + 0.23 * aspect_score
                    + 0.25 * float(crossbar),
                    0.0,
                    1.0,
                )
            )
            if not crossbar or confidence < 0.72:
                continue
            results.append(
                GoalCandidate(
                    left_ground=tuple(map(float, left[0])),
                    right_ground=tuple(map(float, right[0])),
                    left_top=tuple(map(float, left[1])),
                    right_top=tuple(map(float, right[1])),
                    confidence=confidence,
                    crossbar_supported=crossbar,
                    backline_support=backline_support,
                )
            )
    return GoalDetection(tuple(_suppress_goal_duplicates(results)))


def draw_goal_detection(frame: np.ndarray, detection: GoalDetection) -> np.ndarray:
    result = frame.copy()
    for index, candidate in enumerate(detection.candidates, start=1):
        color = (40, 220, 40) if candidate.crossbar_supported else (0, 180, 255)
        polygon = np.asarray(
            [candidate.left_ground, candidate.left_top, candidate.right_top, candidate.right_ground],
            dtype=np.int32,
        )
        cv2.polylines(result, [polygon], False, color, 3, cv2.LINE_AA)
        for point in (candidate.left_ground, candidate.right_ground):
            cv2.circle(result, tuple(np.round(point).astype(int)), 6, (255, 0, 255), -1, cv2.LINE_AA)
        center = tuple(np.round(candidate.center_ground).astype(int))
        cv2.putText(
            result,
            f"G{index}:{candidate.confidence:.2f}",
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return result


def _ordered_post(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    first, second = segment[:2], segment[2:]
    top, ground = (first, second) if first[1] <= second[1] else (second, first)
    return ground, top, float(np.linalg.norm(ground - top))


def _vertical_deviation(segment: np.ndarray) -> float:
    angle = abs(degrees(atan2(segment[3] - segment[1], segment[2] - segment[0]))) % 180.0
    return abs(90.0 - angle)


def _horizontal_deviation(segment: np.ndarray) -> float:
    angle = abs(degrees(atan2(segment[3] - segment[1], segment[2] - segment[0]))) % 180.0
    return min(angle, 180.0 - angle)


def _crossbar_support(
    horizontals: list[np.ndarray],
    left_top: np.ndarray,
    right_top: np.ndarray,
) -> bool:
    expected_y = (left_top[1] + right_top[1]) / 2.0
    for segment in horizontals:
        minimum_x, maximum_x = sorted((segment[0], segment[2]))
        mean_y = (segment[1] + segment[3]) / 2.0
        overlap = min(maximum_x, right_top[0]) - max(minimum_x, left_top[0])
        if overlap >= (right_top[0] - left_top[0]) * 0.45 and abs(mean_y - expected_y) <= 16.0:
            return True
    return False


def _suppress_goal_duplicates(candidates: list[GoalCandidate]) -> list[GoalCandidate]:
    kept: list[GoalCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        center = np.asarray(candidate.center_ground)
        if any(np.linalg.norm(center - np.asarray(other.center_ground)) < 35.0 for other in kept):
            continue
        kept.append(candidate)
    return kept[:5]


def _goal_ground_has_grass(
    grass: np.ndarray,
    left_ground: np.ndarray,
    right_ground: np.ndarray,
) -> bool:
    supports = []
    for point in (left_ground, right_ground):
        x = int(round(point[0]))
        y = int(round(point[1]))
        x1, x2 = max(0, x - 8), min(grass.shape[1], x + 9)
        y1, y2 = max(0, y + 2), min(grass.shape[0], y + 20)
        patch = grass[y1:y2, x1:x2]
        supports.append(float(np.mean(patch > 0)) if patch.size else 0.0)
    return min(supports) >= 0.22


def _backline_support(
    white: np.ndarray,
    left_ground: np.ndarray,
    right_ground: np.ndarray,
) -> float:
    direction = right_ground - left_ground
    start = left_ground - 0.45 * direction
    end = right_ground + 0.45 * direction
    probe = np.zeros_like(white)
    cv2.line(
        probe,
        tuple(np.round(start).astype(int)),
        tuple(np.round(end).astype(int)),
        255,
        5,
        cv2.LINE_AA,
    )
    selected = probe > 0
    return float(np.mean(white[selected] > 0)) if np.any(selected) else 0.0
