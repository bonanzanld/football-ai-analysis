from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

from football_ai.calibration.manual_perspective_reference import ManualPerspectiveView


@dataclass(frozen=True, slots=True)
class ZoomStableSegment:
    start_time_seconds: float
    end_time_seconds: float
    node_count: int

    def __post_init__(self) -> None:
        if self.start_time_seconds > self.end_time_seconds or self.node_count < 1:
            raise ValueError("Ongeldig zoomstabiel segment")

    def contains(self, time_seconds: float, *, boundary_margin_seconds: float = 0.0) -> bool:
        return (
            self.start_time_seconds + boundary_margin_seconds
            <= time_seconds
            <= self.end_time_seconds - boundary_margin_seconds
        )


@dataclass(frozen=True, slots=True)
class StableHorizonEstimate:
    view_label: str
    frame_number: int
    time_seconds: float
    segment_start_seconds: float
    segment_end_seconds: float
    horizon: tuple[float, float, float]


def select_zoom_stable_horizons(
    views: Iterable[ManualPerspectiveView],
    segments: Iterable[ZoomStableSegment],
    *,
    boundary_margin_seconds: float = 1.0,
    minimum_segment_nodes: int = 5,
) -> tuple[StableHorizonEstimate, ...]:
    """Select horizons only from complete views safely inside no-zoom segments."""
    if boundary_margin_seconds < 0.0 or minimum_segment_nodes < 1:
        raise ValueError("Ongeldige stabiele-horizonvoorwaarden")
    stable = tuple(item for item in segments if item.node_count >= minimum_segment_nodes)
    results = []
    for view in views:
        if not view.perspective_complete:
            continue
        containing = next(
            (
                segment for segment in stable
                if segment.contains(
                    view.time_seconds, boundary_margin_seconds=boundary_margin_seconds
                )
            ),
            None,
        )
        if containing is None:
            continue
        results.append(
            StableHorizonEstimate(
                view.label,
                view.frame_number,
                view.time_seconds,
                containing.start_time_seconds,
                containing.end_time_seconds,
                view.horizon(),
            )
        )
    return tuple(results)
