from __future__ import annotations

import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium


class FootballDetector:
    """Detecteert spelers en voetballen met RF-DETR Medium."""

    def __init__(
        self,
        player_threshold: float = 0.20,
        ball_threshold: float = 0.05,
    ) -> None:
        self.player_threshold = player_threshold
        self.ball_threshold = ball_threshold

        print("RF-DETR Medium laden...")
        self.model = RFDETRMedium()
        print("RF-DETR Medium geladen.")

    def detect(
        self,
        frame: np.ndarray,
    ) -> tuple[sv.Detections, sv.Detections, sv.Detections]:
        """
        Verwerk één OpenCV-frame.

        Retourneert:
        - alle detecties;
        - personen;
        - sportballen.
        """

        if frame is None or frame.size == 0:
            raise ValueError("Het aangeleverde frame is leeg.")

        # OpenCV gebruikt BGR; RF-DETR verwacht RGB.
        rgb_frame = frame[:, :, ::-1].copy()

        detection_threshold = min(
            self.player_threshold,
            self.ball_threshold,
        )

        detections = self.model.predict(
            rgb_frame,
            threshold=detection_threshold,
        )

        if len(detections) == 0:
            empty = sv.Detections.empty()
            return empty, empty, empty

        # De nieuwste RF-DETR-output bevat normaal class_name.
        class_names = detections.data.get("class_name")

        if class_names is not None:
            normalized_names = np.char.lower(
                class_names.astype(str)
            )

            player_mask = (
                (normalized_names == "person")
                & (detections.confidence >= self.player_threshold)
            )

            ball_mask = (
                (normalized_names == "sports ball")
                & (detections.confidence >= self.ball_threshold)
            )

        else:
            # COCO-mapping: person=0, sports ball=32.
            player_mask = (
                (detections.class_id == 0)
                & (detections.confidence >= self.player_threshold)
            )

            ball_mask = (
                (detections.class_id == 32)
                & (detections.confidence >= self.ball_threshold)
            )

        return (
            detections,
            detections[player_mask],
            detections[ball_mask],
        )