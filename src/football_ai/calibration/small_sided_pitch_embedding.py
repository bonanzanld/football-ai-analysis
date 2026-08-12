from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from football_ai.calibration.full_pitch_markings import (
    FullPitchMarkingModel,
    PitchMarkingLine,
)
from football_ai.calibration.ground_line_evidence import GroundLineFamily


class FullPitchHalf(str, Enum):
    GOAL_A = "goal_a_half"
    GOAL_B = "goal_b_half"


@dataclass(frozen=True, slots=True)
class PerpendicularHalfPitchEmbedding:
    """Metric placement of a rotated small-sided pitch on one full-pitch half.

    The small pitch length runs across the full-pitch width. Its width runs
    along, and is centered within, one half of the full-pitch length.
    """

    full_length_m: float
    full_width_m: float
    small_length_m: float
    small_width_m: float
    half: FullPitchHalf

    def __post_init__(self) -> None:
        values = (
            self.full_length_m,
            self.full_width_m,
            self.small_length_m,
            self.small_width_m,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Pitch dimensions must be positive")
        if self.small_length_m > self.full_width_m:
            raise ValueError("Small-pitch length does not fit across full-pitch width")
        if self.small_width_m > self.full_length_m / 2.0:
            raise ValueError("Small-pitch width does not fit inside one full-pitch half")

    @property
    def full_sideline_margin_m(self) -> float:
        return (self.full_width_m - self.small_length_m) / 2.0

    @property
    def half_end_margin_m(self) -> float:
        return (self.full_length_m / 2.0 - self.small_width_m) / 2.0

    @property
    def full_length_bounds_m(self) -> tuple[float, float]:
        margin = self.half_end_margin_m
        if self.half is FullPitchHalf.GOAL_A:
            return margin, self.full_length_m / 2.0 - margin
        return self.full_length_m / 2.0 + margin, self.full_length_m - margin

    @property
    def full_width_bounds_m(self) -> tuple[float, float]:
        margin = self.full_sideline_margin_m
        return margin, self.full_width_m - margin

    def small_to_full(self, point: tuple[float, float]) -> tuple[float, float]:
        """Map small (goal-to-goal, touchline-to-touchline) to full coordinates."""

        small_x, small_y = (float(value) for value in point)
        full_x_min, full_x_max = self.full_length_bounds_m
        full_y_min, _full_y_max = self.full_width_bounds_m
        if self.half is FullPitchHalf.GOAL_A:
            full_x = full_x_min + small_y
        else:
            full_x = full_x_max - small_y
        full_y = full_y_min + small_x
        return full_x, full_y

    def full_to_small(self, point: tuple[float, float]) -> tuple[float, float]:
        """Map a full-pitch point into the rotated small-pitch coordinate system."""

        full_x, full_y = (float(value) for value in point)
        full_x_min, full_x_max = self.full_length_bounds_m
        full_y_min, _full_y_max = self.full_width_bounds_m
        small_x = full_y - full_y_min
        small_y = (
            full_x - full_x_min
            if self.half is FullPitchHalf.GOAL_A
            else full_x_max - full_x
        )
        return small_x, small_y

    @property
    def corners_on_full_pitch(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            self.small_to_full(point)
            for point in (
                (0.0, 0.0),
                (self.small_length_m, 0.0),
                (self.small_length_m, self.small_width_m),
                (0.0, self.small_width_m),
            )
        )


def embed_full_pitch_markings(
    model: FullPitchMarkingModel,
    embedding: PerpendicularHalfPitchEmbedding,
    *,
    exterior_margin_m: float = 20.0,
) -> FullPitchMarkingModel:
    """Express intersecting painted 11v11 lines in rotated small-pitch axes."""

    if exterior_margin_m < 0.0:
        raise ValueError("Exterior marking margin cannot be negative")
    if (
        abs(model.pitch_length_m - embedding.full_length_m) > 1e-9
        or abs(model.pitch_width_m - embedding.full_width_m) > 1e-9
    ):
        raise ValueError("Full-pitch marking model and embedding dimensions differ")
    full_x_min, full_x_max = embedding.full_length_bounds_m
    full_y_min, full_y_max = embedding.full_width_bounds_m
    lines: list[PitchMarkingLine] = []
    for line in model.lines:
        if line.family is GroundLineFamily.LONGITUDINAL:
            if not full_y_min - exterior_margin_m <= line.offset_m <= full_y_max + exterior_margin_m:
                continue
            lines.append(
                PitchMarkingLine(
                    line.marking_id,
                    line.label,
                    GroundLineFamily.TRANSVERSE,
                    line.offset_m - full_y_min,
                    min(line.nominal_length_m, embedding.small_width_m),
                )
            )
        else:
            if not full_x_min - exterior_margin_m <= line.offset_m <= full_x_max + exterior_margin_m:
                continue
            offset = (
                line.offset_m - full_x_min
                if embedding.half is FullPitchHalf.GOAL_A
                else full_x_max - line.offset_m
            )
            lines.append(
                PitchMarkingLine(
                    line.marking_id,
                    line.label,
                    GroundLineFamily.LONGITUDINAL,
                    offset,
                    min(line.nominal_length_m, embedding.small_length_m),
                )
            )
    return FullPitchMarkingModel(
        embedding.small_length_m,
        embedding.small_width_m,
        model.center_circle_radius_m,
        model.penalty_area_depth_m,
        model.penalty_area_width_m,
        model.goal_area_depth_m,
        model.goal_area_width_m,
        tuple(lines),
    )
