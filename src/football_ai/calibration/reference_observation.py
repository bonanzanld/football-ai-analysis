from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from football_ai.calibration.reference_3d import FootballFieldReference3D


class ObservationSource(str, Enum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"


@dataclass(frozen=True, slots=True)
class ReferenceObservation2D:
    landmark_id: str
    image_point: tuple[float, float]
    source: ObservationSource = ObservationSource.MANUAL
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Observatieconfidence moet tussen 0 en 1 liggen.")

    def to_dict(self) -> dict:
        return {
            "landmark_id": self.landmark_id,
            "image_point": list(self.image_point),
            "source": self.source.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReferenceObservation2D":
        return cls(
            landmark_id=str(data["landmark_id"]),
            image_point=tuple(map(float, data["image_point"])),
            source=ObservationSource(str(data.get("source", ObservationSource.MANUAL.value))),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass(frozen=True, slots=True)
class CameraViewObservations:
    frame_number: int
    camera_state: int
    observations: tuple[ReferenceObservation2D, ...]

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "camera_state": self.camera_state,
            "observations": [item.to_dict() for item in self.observations],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraViewObservations":
        return cls(
            frame_number=int(data["frame_number"]),
            camera_state=int(data["camera_state"]),
            observations=tuple(
                ReferenceObservation2D.from_dict(item)
                for item in data.get("observations", ())
            ),
        )

    def validate(self, reference: FootballFieldReference3D) -> None:
        identifiers = [item.landmark_id for item in self.observations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Een referentiepunt mag per camerastand maar eenmaal voorkomen.")
        for landmark_id in identifiers:
            reference.landmark(landmark_id)

    def ground_observations(
        self,
        reference: FootballFieldReference3D,
    ) -> tuple[ReferenceObservation2D, ...]:
        return tuple(
            item for item in self.observations
            if reference.landmark(item.landmark_id).is_on_ground
        )

    def elevated_observations(
        self,
        reference: FootballFieldReference3D,
    ) -> tuple[ReferenceObservation2D, ...]:
        return tuple(
            item for item in self.observations
            if not reference.landmark(item.landmark_id).is_on_ground
        )

    def supports_ground_homography(self, reference: FootballFieldReference3D) -> bool:
        ground = self.ground_observations(reference)
        if len(ground) < 4:
            return False
        world_points = np.asarray(
            [reference.landmark(item.landmark_id).point.as_tuple()[:2] for item in ground],
            dtype=np.float32,
        )
        image_points = np.asarray([item.image_point for item in ground], dtype=np.float32)
        world_hull = cv2.convexHull(world_points)
        image_hull = cv2.convexHull(image_points)
        return (
            abs(float(cv2.contourArea(world_hull))) > 1e-6
            and abs(float(cv2.contourArea(image_hull))) > 1e-6
        )

    def supports_3d_pose(self, reference: FootballFieldReference3D) -> bool:
        if len(self.observations) < 6:
            return False
        return (
            len(self.elevated_observations(reference)) >= 2
            and self.supports_ground_homography(reference)
        )
