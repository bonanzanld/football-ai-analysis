from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


VALID_REVIEW_VISIBILITY = {"visible", "occluded", "not_visible"}
VALID_OCCLUSION = {"none", "player", "shadow", "other"}


def centered_image_box(
    point: tuple[int, int],
    *,
    size: float,
    scale: float,
    offset_x: int,
    offset_y: int,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    if scale <= 0.0 or size < 2.0:
        raise ValueError("scale must be positive and size must be at least 2 pixels")
    center_x = (point[0] - offset_x) / scale
    center_y = (point[1] - offset_y) / scale
    if not (0.0 <= center_x < image_width and 0.0 <= center_y < image_height):
        return None
    half = size / 2.0
    return (
        max(0.0, center_x - half),
        max(0.0, center_y - half),
        min(float(image_width), center_x + half),
        min(float(image_height), center_y + half),
    )


def image_box_from_drag(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    scale: float,
    offset_x: int,
    offset_y: int,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    x1, x2 = sorted((start[0], end[0]))
    y1, y2 = sorted((start[1], end[1]))
    x1 = max(0.0, min(float(image_width), (x1 - offset_x) / scale))
    x2 = max(0.0, min(float(image_width), (x2 - offset_x) / scale))
    y1 = max(0.0, min(float(image_height), (y1 - offset_y) / scale))
    y2 = max(0.0, min(float(image_height), (y2 - offset_y) / scale))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return (x1, y1, x2, y2)


def confirm_ball_annotation(
    annotation: dict[str, object],
    *,
    visibility: str,
    box: tuple[float, float, float, float] | None,
    occlusion: str = "none",
) -> dict[str, object]:
    if visibility not in VALID_REVIEW_VISIBILITY:
        raise ValueError(f"Unknown review visibility: {visibility}")
    if occlusion not in VALID_OCCLUSION:
        raise ValueError(f"Unknown occlusion type: {occlusion}")
    if visibility in {"visible", "occluded"} and box is None:
        raise ValueError(f"A {visibility} ball requires a box")
    if visibility == "not_visible":
        box = None
        occlusion = "none"
    elif visibility == "visible" and occlusion == "player":
        occlusion = "none"
    updated = deepcopy(annotation)
    updated["visibility"] = visibility
    updated["ball_box"] = None if box is None else list(box)
    updated["occlusion"] = occlusion
    updated["review_status"] = "human_reviewed"
    return updated


def review_progress(annotations: list[dict[str, object]]) -> tuple[int, int]:
    reviewed = sum(
        item.get("review_status") == "human_reviewed" for item in annotations
    )
    return reviewed, len(annotations)


def save_review_manifest(payload: dict[str, object], path: str | Path) -> None:
    target = Path(path)
    annotations = list(payload.get("annotations", []))
    reviewed, total = review_progress(annotations)
    payload["human_review_complete"] = bool(total and reviewed == total)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
