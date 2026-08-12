from __future__ import annotations

import cv2
import numpy as np


def candidate_temporal_features(
    candidate: dict[str, object],
    candidates_by_frame: dict[int, list[dict[str, object]]],
    *,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Describe label-free candidate continuity across nearby frames.

    Neighbours are selected only by geometry. Human labels are deliberately not
    used, so these features remain valid in clip-separated evaluation and at
    inference time.
    """

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive")
    frame_number = int(candidate["frame_number"])
    box = tuple(float(value) for value in candidate["box"])
    center = np.asarray(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    diagonal = float(np.hypot(frame_width, frame_height))
    confidence = float(candidate["confidence"])
    frame_candidates = candidates_by_frame.get(frame_number, [])
    confidences = sorted(
        (float(item["confidence"]) for item in frame_candidates), reverse=True
    )
    confidence_rank = confidences.index(confidence) if confidence in confidences else 0
    rank_fraction = confidence_rank / max(1, len(confidences) - 1)

    features = [np.log1p(len(frame_candidates)), rank_fraction]
    neighbour_centers: dict[int, np.ndarray] = {}
    for offset in (-2, -1, 1, 2):
        neighbours = candidates_by_frame.get(frame_number + offset, [])
        if not neighbours:
            features.extend((1.0, 1.0, 1.0))
            continue
        neighbour = min(
            neighbours,
            key=lambda item: np.linalg.norm(
                np.asarray(
                    (
                        (float(item["box"][0]) + float(item["box"][2])) / 2.0,
                        (float(item["box"][1]) + float(item["box"][3])) / 2.0,
                    )
                )
                - center
            ),
        )
        neighbour_box = tuple(float(value) for value in neighbour["box"])
        neighbour_center = np.asarray(
            (
                (neighbour_box[0] + neighbour_box[2]) / 2.0,
                (neighbour_box[1] + neighbour_box[3]) / 2.0,
            )
        )
        neighbour_centers[offset] = neighbour_center
        neighbour_area = max(
            1.0,
            (neighbour_box[2] - neighbour_box[0])
            * (neighbour_box[3] - neighbour_box[1]),
        )
        features.extend(
            (
                float(np.linalg.norm(neighbour_center - center) / diagonal),
                min(1.0, abs(float(np.log(neighbour_area / area))) / 4.0),
                abs(float(neighbour["confidence"]) - confidence),
            )
        )

    for distance in (1, 2):
        previous = neighbour_centers.get(-distance)
        following = neighbour_centers.get(distance)
        midpoint_error = (
            float(np.linalg.norm((previous + following) / 2.0 - center) / diagonal)
            if previous is not None and following is not None
            else 1.0
        )
        features.append(midpoint_error)
    return np.asarray(features, dtype=np.float32)


def candidate_patch_features(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    output_size: int = 20,
    context_scale: float = 2.0,
) -> np.ndarray:
    """Return normalized visual features around one detector candidate."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image")
    x1, y1, x2, y2 = (float(value) for value in box)
    width = max(2.0, x2 - x1)
    height = max(2.0, y2 - y1)
    side = max(width, height) * context_scale
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = max(0, int(round(center_x - side / 2.0)))
    top = max(0, int(round(center_y - side / 2.0)))
    right = min(image.shape[1], int(round(center_x + side / 2.0)))
    bottom = min(image.shape[0], int(round(center_y + side / 2.0)))
    if right <= left or bottom <= top:
        raise ValueError("Candidate box falls outside the image")
    patch = cv2.resize(
        image[top:bottom, left:right],
        (output_size, output_size),
        interpolation=cv2.INTER_AREA,
    )
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_32F) / 255.0
    return np.concatenate((lab.reshape(-1), edges.reshape(-1)))
