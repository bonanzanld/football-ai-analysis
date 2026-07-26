from __future__ import annotations

from dataclasses import dataclass
from math import degrees

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class DetectedFieldLine:
    line_key: int
    points: tuple[tuple[float, float], ...]
    confidence: float


def detect_goal_end_field_lines(
    image: np.ndarray,
    far_corner: tuple[float, float],
    near_corner: tuple[float, float],
    goal_line_key: int,
    far_sideline_key: int = 3,
    near_sideline_key: int = 4,
) -> tuple[DetectedFieldLine, ...]:
    """Detecteer achterlijn en beide zijlijnrichtingen vanuit twee hoeken."""
    far = np.asarray(far_corner, dtype=np.float64)
    near = np.asarray(near_corner, dtype=np.float64)
    back_direction = near - far
    if np.linalg.norm(back_direction) < 20.0:
        raise ValueError("De twee veldhoeken liggen te dicht bij elkaar.")

    back_points = tuple(
        tuple(map(float, far + alpha * back_direction))
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    candidates = _white_line_candidates(image)
    far_line = _best_sideline(candidates, far, back_direction)
    near_line = _best_sideline(candidates, near, back_direction)
    if far_line is None or near_line is None:
        raise ValueError(
            "Automatische zijlijndetectie vond niet bij beide hoeken een "
            "betrouwbare witte lijn. Kies een scherper doelbeeld."
        )
    return (
        DetectedFieldLine(goal_line_key, back_points, 1.0),
        DetectedFieldLine(far_sideline_key, far_line[0], far_line[1]),
        DetectedFieldLine(near_sideline_key, near_line[0], near_line[1]),
    )


def _white_line_candidates(image: np.ndarray) -> list[np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 125), (180, 105, 255))
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    edges = cv2.Canny(white, 50, 150)
    minimum_length = max(35, int(min(image.shape[:2]) * 0.06))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720.0,
        threshold=25,
        minLineLength=minimum_length,
        maxLineGap=28,
    )
    if lines is None:
        return []
    return [line.reshape(4).astype(np.float64) for line in lines]


def _best_sideline(
    candidates: list[np.ndarray],
    corner: np.ndarray,
    back_direction: np.ndarray,
) -> tuple[tuple[tuple[float, float], ...], float] | None:
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    back_unit = back_direction / np.linalg.norm(back_direction)
    for candidate in candidates:
        first, second = candidate[:2], candidate[2:]
        direction = second - first
        length = float(np.linalg.norm(direction))
        if length < 30.0:
            continue
        unit = direction / length
        angle = degrees(np.arccos(np.clip(abs(unit @ back_unit), 0.0, 1.0)))
        if angle < 12.0:
            continue
        offset = corner - first
        cross_value = direction[0] * offset[1] - direction[1] * offset[0]
        distance = abs(float(cross_value)) / length
        if distance > 32.0:
            continue
        score = length * (angle / 90.0) / (1.0 + distance / 8.0)
        if best is None or score > best[0]:
            best = (score, first, second)
    if best is None:
        return None
    score, first, second = best
    direction = second - first
    projection = first + direction * (
        float((corner - first) @ direction) / float(direction @ direction)
    )
    points = (
        tuple(map(float, projection)),
        tuple(map(float, first)),
        tuple(map(float, second)),
    )
    confidence = min(1.0, score / 180.0)
    return points, confidence
