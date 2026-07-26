from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from football_ai.calibration.camera_anchor_bank_3d import CameraAnchorBank3D
from football_ai.calibration.camera_anchor_recognition import AnchorRecognition, CameraAnchorRecognizer
from football_ai.calibration.camera_projection_3d import CameraProjection3D
from football_ai.calibration.local_anchor_projection import LocalProjectionResult, estimate_local_anchor_projection
from football_ai.calibration.reference_3d import FootballFieldReference3D


@dataclass(frozen=True, slots=True)
class RuntimeFieldProjection:
    valid: bool
    anchor_id: str | None
    projection: CameraProjection3D | None
    recognition: AnchorRecognition
    local: LocalProjectionResult | None
    reason: str


class CameraAnchorRuntime:
    """Resolve each frame directly against immutable camera anchors."""

    def __init__(
        self,
        bank: CameraAnchorBank3D,
        reference: FootballFieldReference3D,
        anchor_frames: dict[str, np.ndarray],
    ) -> None:
        missing = {item.anchor_id for item in bank.anchors} - anchor_frames.keys()
        if missing:
            raise ValueError(f"Ankerbeelden ontbreken: {', '.join(sorted(missing))}")
        self.bank = bank
        self.reference = reference
        self.anchor_frames = anchor_frames
        self.anchor_by_id = {item.anchor_id: item for item in bank.anchors}
        self.recognizer = CameraAnchorRecognizer.from_frames(anchor_frames)

    def project(self, frame: np.ndarray) -> RuntimeFieldProjection:
        recognition = self.recognizer.recognize(frame)
        if recognition.anchor_id is None:
            return RuntimeFieldProjection(False, None, None, recognition, None, recognition.reason)
        anchor = self.anchor_by_id[recognition.anchor_id]
        local = estimate_local_anchor_projection(
            self.anchor_frames[anchor.anchor_id],
            frame,
            anchor.projection,
            self.reference,
        )
        if not local.valid or local.projection is None:
            return RuntimeFieldProjection(False, None, None, recognition, local, local.reason)
        return RuntimeFieldProjection(
            True,
            anchor.anchor_id,
            local.projection,
            recognition,
            local,
            local.reason,
        )

    def project_with_anchor(
        self,
        frame: np.ndarray,
        anchor_id: str,
        recognition: AnchorRecognition,
    ) -> RuntimeFieldProjection:
        """Project through an explicitly selected anchor during offline analysis.

        The ordinary real-time path deliberately rejects ambiguous anchor
        recognition.  An offline analyzer can use temporal context to select
        one of those candidates and then validate the local geometry here.
        """
        if anchor_id not in self.anchor_by_id:
            raise ValueError(f"Onbekend camera-anker: {anchor_id}")
        anchor = self.anchor_by_id[anchor_id]
        local = estimate_local_anchor_projection(
            self.anchor_frames[anchor_id],
            frame,
            anchor.projection,
            self.reference,
        )
        if not local.valid or local.projection is None:
            return RuntimeFieldProjection(False, anchor_id, None, recognition, local, local.reason)
        return RuntimeFieldProjection(
            True,
            anchor_id,
            local.projection,
            recognition,
            local,
            local.reason,
        )
