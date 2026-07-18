from __future__ import annotations

import numpy as np


class SizeFilter:
    """
    Controleert of een bounding box qua afmetingen en verhouding
    plausibel is voor een speler.
    """

    def __init__(
        self,
        minimum_box_height: int = 24,
        minimum_aspect_ratio: float = 1.15,
        maximum_aspect_ratio: float = 6.0,
        minimum_foot_y_ratio: float = 0.15,
    ) -> None:
        self.minimum_box_height = minimum_box_height
        self.minimum_aspect_ratio = minimum_aspect_ratio
        self.maximum_aspect_ratio = maximum_aspect_ratio
        self.minimum_foot_y_ratio = minimum_foot_y_ratio

    def accept(
        self,
        bounding_box: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        x1, y1, x2, y2 = bounding_box.astype(float)

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            return False

        if box_height < self.minimum_box_height:
            return False

        aspect_ratio = box_height / box_width

        if not (
            self.minimum_aspect_ratio
            <= aspect_ratio
            <= self.maximum_aspect_ratio
        ):
            return False

        if frame_height <= 0:
            return False

        foot_y_ratio = y2 / frame_height

        return foot_y_ratio >= self.minimum_foot_y_ratio
