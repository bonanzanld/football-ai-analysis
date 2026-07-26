from __future__ import annotations

from enum import Enum


class FieldZone(str, Enum):
    INSIDE = "binnen"
    EDGE = "randzone"
    OUTSIDE = "buiten"


def classify_field_position(
    point: tuple[float, float],
    pitch_length_m: float,
    pitch_width_m: float,
    edge_margin_m: float = 1.5,
) -> FieldZone:
    """Classify a pitch-space point with a forgiving band around every edge."""
    if pitch_length_m <= 0.0 or pitch_width_m <= 0.0:
        raise ValueError("Veldafmetingen moeten positief zijn.")
    if edge_margin_m < 0.0:
        raise ValueError("Randmarge mag niet negatief zijn.")
    x, y = point
    if not (-edge_margin_m <= x <= pitch_length_m + edge_margin_m):
        return FieldZone.OUTSIDE
    if not (-edge_margin_m <= y <= pitch_width_m + edge_margin_m):
        return FieldZone.OUTSIDE
    if edge_margin_m <= x <= pitch_length_m - edge_margin_m and edge_margin_m <= y <= pitch_width_m - edge_margin_m:
        return FieldZone.INSIDE
    return FieldZone.EDGE
