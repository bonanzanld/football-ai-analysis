from __future__ import annotations

import cv2
import numpy as np


class GreenFilter:
    """
    Controleert hoeveel groen veld zichtbaar is rond de voeten.
    """

    def __init__(
        self,
        minimum_green_ratio: float = 0.18,
    ) -> None:
        if not 0.0 <= minimum_green_ratio <= 1.0:
            raise ValueError(
                "minimum_green_ratio moet tussen 0 en 1 liggen."
            )

        self.minimum_green_ratio = minimum_green_ratio

    def accept(
        self,
        frame: np.ndarray,
        bounding_box: np.ndarray,
    ) -> bool:
        return (
            self.calculate_green_ratio(
                frame=frame,
                bounding_box=bounding_box,
            )
            >= self.minimum_green_ratio
        )

    def calculate_green_ratio(
        self,
        frame: np.ndarray,
        bounding_box: np.ndarray,
    ) -> float:
        frame_height, frame_width = frame.shape[:2]

        x1, y1, x2, y2 = bounding_box.astype(int)

        box_width = x2 - x1
        box_height = y2 - y1
        center_x = int((x1 + x2) / 2)

        patch_half_width = max(
            4,
            int(box_width * 0.40),
        )
        patch_height_above = max(
            3,
            int(box_height * 0.08),
        )
        patch_height_below = max(
            4,
            int(box_height * 0.12),
        )

        patch_x1 = max(0, center_x - patch_half_width)
        patch_x2 = min(frame_width, center_x + patch_half_width)
        patch_y1 = max(0, y2 - patch_height_above)
        patch_y2 = min(frame_height, y2 + patch_height_below)

        if patch_x2 <= patch_x1 or patch_y2 <= patch_y1:
            return 0.0

        foot_patch = frame[
            patch_y1:patch_y2,
            patch_x1:patch_x2,
        ]

        if foot_patch.size == 0:
            return 0.0

        hsv_patch = cv2.cvtColor(
            foot_patch,
            cv2.COLOR_BGR2HSV,
        )

        lower_green = np.array(
            [28, 25, 25],
            dtype=np.uint8,
        )
        upper_green = np.array(
            [100, 255, 255],
            dtype=np.uint8,
        )

        green_mask = cv2.inRange(
            hsv_patch,
            lower_green,
            upper_green,
        )

        total_pixels = int(green_mask.size)

        if total_pixels == 0:
            return 0.0

        green_pixels = int(
            cv2.countNonZero(green_mask)
        )

        return green_pixels / total_pixels
