from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from football_ai.calibration.bootstrap.detection_profile import PitchDetectionProfile


class LandmarkKind(str, Enum):
    FIELD_CORNER = "field_corner"
    GOAL_POST_BOTTOM = "goal_post_bottom"
    GOAL_POST_TOP = "goal_post_top"
    GROUND_POINT = "ground_point"


@dataclass(frozen=True, slots=True)
class Point3D:
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass(frozen=True, slots=True)
class ReferenceLandmark3D:
    landmark_id: str
    label: str
    kind: LandmarkKind
    point: Point3D

    @property
    def is_on_ground(self) -> bool:
        return abs(self.point.z) < 1e-9


@dataclass(frozen=True, slots=True)
class FootballFieldReference3D:
    match_format: str
    pitch_length_m: float
    pitch_width_m: float
    goal_width_m: float
    goal_height_m: float
    landmarks: tuple[ReferenceLandmark3D, ...]

    def landmark(self, landmark_id: str) -> ReferenceLandmark3D:
        for item in self.landmarks:
            if item.landmark_id == landmark_id:
                return item
        raise KeyError(f"Onbekend 3D-referentiepunt: {landmark_id}")

    @property
    def ground_landmarks(self) -> tuple[ReferenceLandmark3D, ...]:
        return tuple(item for item in self.landmarks if item.is_on_ground)

    @property
    def elevated_landmarks(self) -> tuple[ReferenceLandmark3D, ...]:
        return tuple(item for item in self.landmarks if not item.is_on_ground)


def create_field_reference_3d(profile: PitchDetectionProfile) -> FootballFieldReference3D:
    """Create one immutable metric field coordinate system.

    x runs from the left goal (A) to the right goal (B), y from the far
    sideline to the camera-side sideline, and z vertically upwards.
    """
    length = profile.pitch_length_m
    width = profile.pitch_width_m
    half_goal = profile.goal_width_m / 2.0
    goal_rear_y = width / 2.0 - half_goal
    goal_front_y = width / 2.0 + half_goal
    landmarks = (
        _landmark("corner_a_rear", "Veldhoek links/ver", LandmarkKind.FIELD_CORNER, 0.0, 0.0, 0.0),
        _landmark("corner_a_front", "Veldhoek links/camera", LandmarkKind.FIELD_CORNER, 0.0, width, 0.0),
        _landmark("corner_b_rear", "Veldhoek rechts/ver", LandmarkKind.FIELD_CORNER, length, 0.0, 0.0),
        _landmark("corner_b_front", "Veldhoek rechts/camera", LandmarkKind.FIELD_CORNER, length, width, 0.0),
        _landmark("goal_a_rear_bottom", "Doel A verste paal onder", LandmarkKind.GOAL_POST_BOTTOM, 0.0, goal_rear_y, 0.0),
        _landmark("goal_a_front_bottom", "Doel A nabije paal onder", LandmarkKind.GOAL_POST_BOTTOM, 0.0, goal_front_y, 0.0),
        _landmark("goal_a_rear_top", "Doel A verste paal boven", LandmarkKind.GOAL_POST_TOP, 0.0, goal_rear_y, profile.goal_height_m),
        _landmark("goal_a_front_top", "Doel A nabije paal boven", LandmarkKind.GOAL_POST_TOP, 0.0, goal_front_y, profile.goal_height_m),
        _landmark("goal_b_rear_bottom", "Doel B verste paal onder", LandmarkKind.GOAL_POST_BOTTOM, length, goal_rear_y, 0.0),
        _landmark("goal_b_front_bottom", "Doel B nabije paal onder", LandmarkKind.GOAL_POST_BOTTOM, length, goal_front_y, 0.0),
        _landmark("goal_b_rear_top", "Doel B verste paal boven", LandmarkKind.GOAL_POST_TOP, length, goal_rear_y, profile.goal_height_m),
        _landmark("goal_b_front_top", "Doel B nabije paal boven", LandmarkKind.GOAL_POST_TOP, length, goal_front_y, profile.goal_height_m),
        _landmark("midline_rear", "Middenlijn op verste zijlijn", LandmarkKind.GROUND_POINT, length / 2.0, 0.0, 0.0),
        _landmark("midline_front", "Middenlijn op nabije zijlijn", LandmarkKind.GROUND_POINT, length / 2.0, width, 0.0),
        _landmark("center_spot", "Middenstip", LandmarkKind.GROUND_POINT, length / 2.0, width / 2.0, 0.0),
    )
    return FootballFieldReference3D(
        match_format=profile.match_format.value,
        pitch_length_m=length,
        pitch_width_m=width,
        goal_width_m=profile.goal_width_m,
        goal_height_m=profile.goal_height_m,
        landmarks=landmarks,
    )


def _landmark(
    landmark_id: str,
    label: str,
    kind: LandmarkKind,
    x: float,
    y: float,
    z: float,
) -> ReferenceLandmark3D:
    return ReferenceLandmark3D(landmark_id, label, kind, Point3D(x, y, z))
