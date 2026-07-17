from __future__ import annotations

import cv2
import numpy as np
import supervision as sv


class PlayerFilter:
    """
    Filtert person-detecties die waarschijnlijk geen spelers
    op het voetbalveld zijn.

    De eerste versie gebruikt:
    - minimale bounding-boxafmetingen;
    - verhouding tussen hoogte en breedte;
    - positie van het voetpunt;
    - hoeveelheid groen veld rond de voeten.
    """

    def __init__(
        self,
        minimum_box_height: int = 24,
        minimum_aspect_ratio: float = 1.15,
        maximum_aspect_ratio: float = 6.0,
        minimum_foot_y_ratio: float = 0.15,
        minimum_green_ratio: float = 0.18,
    ) -> None:
        self.minimum_box_height = minimum_box_height
        self.minimum_aspect_ratio = minimum_aspect_ratio
        self.maximum_aspect_ratio = maximum_aspect_ratio
        self.minimum_foot_y_ratio = minimum_foot_y_ratio
        self.minimum_green_ratio = minimum_green_ratio

    def filter(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
    ) -> sv.Detections:
        """
        Geeft alleen detecties terug die waarschijnlijk spelers
        op het veld zijn.
        """
        if len(detections) == 0:
            return detections

        keep_mask = np.zeros(
            len(detections),
            dtype=bool,
        )

        for index in range(len(detections)):
            bounding_box = detections.xyxy[index]

            keep_mask[index] = self._is_likely_player(
                frame=frame,
                bounding_box=bounding_box,
            )

        return detections[keep_mask]

    def _is_likely_player(
        self,
        frame: np.ndarray,
        bounding_box: np.ndarray,
    ) -> bool:
        frame_height, frame_width = frame.shape[:2]

        x1, y1, x2, y2 = bounding_box.astype(int)

        x1 = max(
            0,
            min(x1, frame_width - 1),
        )
        x2 = max(
            0,
            min(x2, frame_width),
        )
        y1 = max(
            0,
            min(y1, frame_height - 1),
        )
        y2 = max(
            0,
            min(y2, frame_height),
        )

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            return False

        if box_height < self.minimum_box_height:
            return False

        aspect_ratio = box_height / box_width

        if aspect_ratio < self.minimum_aspect_ratio:
            return False

        if aspect_ratio > self.maximum_aspect_ratio:
            return False

        foot_y_ratio = y2 / frame_height

        if foot_y_ratio < self.minimum_foot_y_ratio:
            return False

        green_ratio = self._calculate_green_ratio_near_feet(
            frame=frame,
            bounding_box=np.array(
                [x1, y1, x2, y2],
                dtype=np.float32,
            ),
        )

        if green_ratio < self.minimum_green_ratio:
            return False

        return True

    def _calculate_green_ratio_near_feet(
        self,
        frame: np.ndarray,
        bounding_box: np.ndarray,
    ) -> float:
        """
        Berekent hoeveel groen veld zichtbaar is in een klein
        gebied rondom en direct onder de voeten.
        """
        frame_height, frame_width = frame.shape[:2]

        x1, y1, x2, y2 = bounding_box.astype(int)

        box_width = x2 - x1
        box_height = y2 - y1

        center_x = int(
            (x1 + x2) / 2
        )

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

        patch_x1 = max(
            0,
            center_x - patch_half_width,
        )
        patch_x2 = min(
            frame_width,
            center_x + patch_half_width,
        )

        patch_y1 = max(
            0,
            y2 - patch_height_above,
        )
        patch_y2 = min(
            frame_height,
            y2 + patch_height_below,
        )

        if (
            patch_x2 <= patch_x1
            or patch_y2 <= patch_y1
        ):
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

        green_pixels = int(
            cv2.countNonZero(green_mask)
        )

        total_pixels = int(
            green_mask.size
        )

        if total_pixels == 0:
            return 0.0

        return green_pixels / total_pixels