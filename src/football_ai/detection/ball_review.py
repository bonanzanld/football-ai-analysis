from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import cv2

from football_ai.tracking.online_camera_motion import transform_box


VALID_REVIEW_VISIBILITY = {"visible", "occluded", "not_visible"}
VALID_OCCLUSION = {"none", "player", "shadow", "other"}


def visible_human_review_indices(
    annotations: Sequence[Mapping[str, object]],
) -> list[int]:
    """Return the stable queue for manually tightening visible-ball boxes."""

    return [
        index
        for index, annotation in enumerate(annotations)
        if annotation.get("review_status") == "human_reviewed"
        and annotation.get("visibility") == "visible"
        and isinstance(annotation.get("ball_box"), Sequence)
        and len(annotation["ball_box"]) == 4
    ]


def propagate_ball_box_optical_flow(
    images: Sequence[np.ndarray],
    *,
    seed_index: int,
    seed_box: Sequence[float],
    maximum_error: float = 18.0,
    maximum_forward_backward_error: float = 1.5,
) -> list[tuple[float, float, float, float] | None]:
    """Propagate one reviewed ball box without pretending the result is ground truth."""

    if not 0 <= seed_index < len(images):
        raise ValueError("seed_index is outside the image sequence")
    if len(seed_box) != 4:
        raise ValueError("seed_box must contain four coordinates")
    x1, y1, x2, y2 = (float(value) for value in seed_box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("seed_box must have positive area")
    results: list[tuple[float, float, float, float] | None] = [None] * len(images)
    results[seed_index] = (x1, y1, x2, y2)
    point = np.asarray([[[(x1 + x2) / 2.0, (y1 + y2) / 2.0]]], dtype=np.float32)
    width, height = x2 - x1, y2 - y1
    previous = cv2.cvtColor(images[seed_index], cv2.COLOR_BGR2GRAY)
    options = {
        "winSize": (31, 31),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    for index in range(seed_index + 1, len(images)):
        current = cv2.cvtColor(images[index], cv2.COLOR_BGR2GRAY)
        following, status, error = cv2.calcOpticalFlowPyrLK(
            previous, current, point, None, **options
        )
        if following is None or status is None or not bool(status[0, 0]):
            break
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current, previous, following, None, **options
        )
        if backward is None or backward_status is None or not bool(backward_status[0, 0]):
            break
        forward_backward_error = float(np.linalg.norm(point - backward))
        tracking_error = float(error[0, 0]) if error is not None else float("inf")
        center_x, center_y = (float(value) for value in following[0, 0])
        image_height, image_width = current.shape[:2]
        if (
            tracking_error > maximum_error
            or forward_backward_error > maximum_forward_backward_error
            or not (0.0 <= center_x < image_width and 0.0 <= center_y < image_height)
        ):
            break
        results[index] = (
            max(0.0, center_x - width / 2.0),
            max(0.0, center_y - height / 2.0),
            min(float(image_width), center_x + width / 2.0),
            min(float(image_height), center_y + height / 2.0),
        )
        point = following
        previous = current
    return results


def proposed_candidate_box(
    frame: Mapping[str, object],
    *,
    image_width: int,
    image_height: int,
    minimum_confidence: float = 0.75,
) -> tuple[float, float, float, float] | None:
    """Return a strong cached candidate in original image space, if available."""

    candidates = frame.get("candidates")
    if not isinstance(candidates, Sequence) or not candidates:
        return None
    valid = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and isinstance(candidate.get("box"), Sequence)
        and len(candidate["box"]) == 4
    ]
    if not valid:
        return None
    selected = max(valid, key=lambda item: float(item.get("confidence", 0.0)))
    if float(selected.get("confidence", 0.0)) < minimum_confidence:
        return None
    transform = np.asarray(frame.get("transform", np.eye(3)), dtype=np.float64)
    if transform.shape != (3, 3):
        return None
    try:
        reference_to_image = np.linalg.inv(transform)
    except np.linalg.LinAlgError:
        return None
    box = transform_box(
        tuple(float(value) for value in selected["box"]),
        reference_to_image,
    )
    x1, y1, x2, y2 = box
    clipped = (
        max(0.0, min(float(image_width), x1)),
        max(0.0, min(float(image_height), y1)),
        max(0.0, min(float(image_width), x2)),
        max(0.0, min(float(image_height), y2)),
    )
    if clipped[2] - clipped[0] < 2.0 or clipped[3] - clipped[1] < 2.0:
        return None
    return clipped


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


def next_unreviewed_index(
    annotations: Sequence[Mapping[str, object]],
    *,
    after: int = -1,
) -> int | None:
    """Find the next open review item, wrapping once after the current item."""

    if not annotations:
        return None
    start = min(max(after + 1, 0), len(annotations))
    indices = (*range(start, len(annotations)), *range(0, start))
    return next(
        (
            index
            for index in indices
            if annotations[index].get("review_status") != "human_reviewed"
        ),
        None,
    )


def next_required_review_index(
    annotations: Sequence[Mapping[str, object]],
    *,
    after: int = -1,
) -> int | None:
    """Return only genuinely open frames or AI drafts marked as checkpoints."""

    if not annotations:
        return None
    start = min(max(after + 1, 0), len(annotations))
    indices = (*range(start, len(annotations)), *range(0, start))
    return next(
        (
            index
            for index in indices
            if annotations[index].get("review_status") == "unreviewed"
            or (
                annotations[index].get("review_status") == "ai_draft"
                and annotations[index].get("review_priority") == "required"
            )
        ),
        None,
    )


def save_review_manifest(payload: dict[str, object], path: str | Path) -> None:
    target = Path(path)
    annotations = list(payload.get("annotations", []))
    reviewed, total = review_progress(annotations)
    payload["human_review_complete"] = bool(total and reviewed == total)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
