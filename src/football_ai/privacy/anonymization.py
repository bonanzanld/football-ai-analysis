from __future__ import annotations

import cv2
import numpy as np


def anonymize_people_heads(
    frame: np.ndarray,
    person_boxes: np.ndarray,
    *,
    head_height_ratio: float = 0.30,
    horizontal_padding_ratio: float = 0.12,
    minimum_kernel: int = 9,
) -> np.ndarray:
    """Blur every detected head region for privacy-safe rendered exports."""
    if not 0.15 <= head_height_ratio <= 0.5:
        raise ValueError("Head height ratio must be between 0.15 and 0.5")
    if horizontal_padding_ratio < 0:
        raise ValueError("Horizontal padding ratio cannot be negative")
    result = frame.copy()
    height, width = result.shape[:2]
    for box in np.asarray(person_boxes, dtype=np.float64).reshape(-1, 4):
        x1, y1, x2, y2 = box
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        if box_width < 2 or box_height < 4:
            continue
        padding = horizontal_padding_ratio * box_width
        left = int(np.clip(np.floor(x1 - padding), 0, width))
        right = int(np.clip(np.ceil(x2 + padding), 0, width))
        top = int(np.clip(np.floor(y1), 0, height))
        bottom = int(np.clip(np.ceil(y1 + head_height_ratio * box_height), 0, height))
        if right <= left or bottom <= top:
            continue
        region = result[top:bottom, left:right]
        kernel = _odd_kernel(max(minimum_kernel, int(round(min(region.shape[:2]) * 0.75))))
        result[top:bottom, left:right] = cv2.GaussianBlur(region, (kernel, kernel), 0)
    return result


def _odd_kernel(value: int) -> int:
    return max(3, value if value % 2 else value + 1)
