from __future__ import annotations

from dataclasses import dataclass

from football_ai.calibration.bootstrap.detection_profile import PitchDetectionProfile
from football_ai.calibration.bootstrap.goal_seed import (
    GoalSeed,
    estimate_backline_endpoints,
)


@dataclass(frozen=True, slots=True)
class SeededFieldCorner:
    name: str
    pitch_point: tuple[float, float]
    normalized_pitch_point: tuple[float, float]
    image_point: tuple[float, float]
    frame_number: int
    measured: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pitch_point": list(self.pitch_point),
            "normalized_pitch_point": list(self.normalized_pitch_point),
            "image_point": list(self.image_point),
            "frame_number": self.frame_number,
            "measured": self.measured,
        }


@dataclass(frozen=True, slots=True)
class SeededFieldContour:
    match_format: str
    pitch_length_m: float
    pitch_width_m: float
    pitch_length_bounds_m: tuple[float, float]
    pitch_width_bounds_m: tuple[float, float]
    corners: tuple[SeededFieldCorner, ...]

    BOUNDARIES = (
        ("achterlijn_links", "linksachter", "linksvoor"),
        ("zijlijn_achter", "linksachter", "rechtsachter"),
        ("achterlijn_rechts", "rechtsachter", "rechtsvoor"),
        ("zijlijn_voor", "linksvoor", "rechtsvoor"),
    )

    def to_dict(self) -> dict:
        return {
            "match_format": self.match_format,
            "pitch_length_m": self.pitch_length_m,
            "pitch_width_m": self.pitch_width_m,
            "metric_dimensions_are_nominal": not (
                self.pitch_length_bounds_m[0] == self.pitch_length_bounds_m[1]
                and self.pitch_width_bounds_m[0] == self.pitch_width_bounds_m[1]
            ),
            "pitch_length_bounds_m": list(self.pitch_length_bounds_m),
            "pitch_width_bounds_m": list(self.pitch_width_bounds_m),
            "corners": [corner.to_dict() for corner in self.corners],
            "boundaries": [
                {"name": name, "from": start, "to": end}
                for name, start, end in self.BOUNDARIES
            ],
        }


def build_seeded_field_contour(
    seeds: tuple[GoalSeed, GoalSeed],
    profile: PitchDetectionProfile,
) -> SeededFieldContour:
    pitch_length_m = profile.pitch_length_m
    pitch_width_m = profile.pitch_width_m
    by_goal = {seed.goal_id: seed for seed in seeds}
    if set(by_goal) != {"A", "B"}:
        raise ValueError("Veldcontour vereist precies Doel A en Doel B.")
    goal_a, goal_b = by_goal["A"], by_goal["B"]
    left_rear, left_front = estimate_backline_endpoints(
        goal_a.first_ground, goal_a.second_ground, goal_a.goal_width_m,
        pitch_width_m, goal_a.rear_corner, goal_a.front_corner,
    )
    right_rear, right_front = estimate_backline_endpoints(
        goal_b.first_ground, goal_b.second_ground, goal_b.goal_width_m,
        pitch_width_m, goal_b.rear_corner, goal_b.front_corner,
    )
    corners = (
        SeededFieldCorner("linksachter", (0.0, 0.0), (0.0, 0.0), left_rear, goal_a.frame_number, goal_a.rear_corner is not None),
        SeededFieldCorner("linksvoor", (0.0, pitch_width_m), (0.0, 1.0), left_front, goal_a.frame_number, goal_a.front_corner is not None),
        SeededFieldCorner("rechtsachter", (pitch_length_m, 0.0), (1.0, 0.0), right_rear, goal_b.frame_number, goal_b.rear_corner is not None),
        SeededFieldCorner("rechtsvoor", (pitch_length_m, pitch_width_m), (1.0, 1.0), right_front, goal_b.frame_number, goal_b.front_corner is not None),
    )
    return SeededFieldContour(
        profile.match_format.value,
        pitch_length_m,
        pitch_width_m,
        (profile.minimum_pitch_length_m, profile.maximum_pitch_length_m),
        (profile.minimum_pitch_width_m, profile.maximum_pitch_width_m),
        corners,
    )
