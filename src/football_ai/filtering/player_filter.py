from __future__ import annotations

import numpy as np
import supervision as sv

from football_ai.filtering.green_filter import GreenFilter
from football_ai.filtering.pitch_filter import PitchFilter
from football_ai.filtering.size_filter import SizeFilter
from football_ai.pitch.calibration_model import PitchCalibration


class PlayerFilter:
    """
    Centrale orchestrator voor de detectiefilters.
    """

    def __init__(
        self,
        minimum_box_height: int = 24,
        minimum_aspect_ratio: float = 1.15,
        maximum_aspect_ratio: float = 6.0,
        minimum_foot_y_ratio: float = 0.15,
        minimum_green_ratio: float = 0.18,
        pitch_calibration: PitchCalibration | None = None,
        pitch_foot_margin_m: float = 0.5,
        pitch_support_margin_m: float = 1.0,
        minimum_pitch_support_ratio: float = 2.0 / 3.0,
    ) -> None:
        self.size_filter = SizeFilter(
            minimum_box_height=minimum_box_height,
            minimum_aspect_ratio=minimum_aspect_ratio,
            maximum_aspect_ratio=maximum_aspect_ratio,
            minimum_foot_y_ratio=minimum_foot_y_ratio,
        )

        self.green_filter = GreenFilter(
            minimum_green_ratio=minimum_green_ratio,
        )

        self.pitch_filter = (
            PitchFilter(
                calibration=pitch_calibration,
                foot_margin_m=pitch_foot_margin_m,
                support_margin_m=pitch_support_margin_m,
                minimum_support_ratio=(
                    minimum_pitch_support_ratio
                ),
            )
            if pitch_calibration is not None
            else None
        )

    def filter(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
    ) -> sv.Detections:
        if len(detections) == 0:
            return detections

        frame_height, frame_width = frame.shape[:2]

        keep_mask = np.zeros(
            len(detections),
            dtype=bool,
        )

        for index, bounding_box in enumerate(
            detections.xyxy
        ):
            keep_mask[index] = self._accept(
                frame=frame,
                bounding_box=bounding_box,
                frame_width=frame_width,
                frame_height=frame_height,
            )

        return detections[keep_mask]

    def _accept(
        self,
        frame: np.ndarray,
        bounding_box: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        clipped_box = self._clip_box(
            bounding_box=bounding_box,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        if clipped_box is None:
            return False

        if not self.size_filter.accept(
            bounding_box=clipped_box,
            frame_width=frame_width,
            frame_height=frame_height,
        ):
            return False

        if not self.green_filter.accept(
            frame=frame,
            bounding_box=clipped_box,
        ):
            return False

        if (
            self.pitch_filter is not None
            and not self.pitch_filter.accept(
                bounding_box=clipped_box,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        ):
            return False

        return True

    def _clip_box(
        self,
        bounding_box: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = bounding_box.astype(float)

        x1 = max(0.0, min(x1, frame_width - 1.0))
        x2 = max(0.0, min(x2, float(frame_width)))
        y1 = max(0.0, min(y1, frame_height - 1.0))
        y2 = max(0.0, min(y2, float(frame_height)))

        if x2 <= x1 or y2 <= y1:
            return None

        return np.array(
            [x1, y1, x2, y2],
            dtype=np.float32,
        )
