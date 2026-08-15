from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from football_ai.calibration.bootstrap.white_line_detection import WhiteLineCandidate
from football_ai.calibration.perspective_parallelism import estimate_vanishing_point_from_lines


@dataclass(frozen=True, slots=True)
class ImageLinePerspective:
    valid: bool
    vanishing_point: tuple[float, float] | None
    supporting_lines: tuple[WhiteLineCandidate, ...]
    rms_degrees: float
    reason: str


def estimate_sideline_perspective(
    candidates: tuple[WhiteLineCandidate, ...],
    polygon: np.ndarray,
    frame_size: tuple[int, int],
    minimum_length_pixels: float = 70.0,
    maximum_residual_degrees: float = 7.0,
    maximum_predicted_residual_degrees: float = 4.0,
) -> ImageLinePerspective:
    """Estimate the shared 11v11/8v8 longitudinal VP in image space."""
    corners = np.asarray(polygon, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ValueError("Vier geprojecteerde veldhoeken vereist.")
    try:
        predicted = np.asarray(
            estimate_vanishing_point_from_lines(
                (
                    (tuple(corners[0]), tuple(corners[1])),
                    (tuple(corners[3]), tuple(corners[2])),
                )
            ),
            dtype=np.float64,
        )
    except ValueError as error:
        return ImageLinePerspective(False, None, (), float("inf"), str(error))
    eligible = tuple(
        item
        for item in candidates
        if item.length_pixels >= minimum_length_pixels
        and item.visual_confidence >= 0.58
        and _residual_to_point(item, predicted) <= 14.0
    )
    predicted_support = tuple(
        item for item in eligible
        if _residual_to_point(item, predicted) <= maximum_predicted_residual_degrees
    )
    if len(eligible) < 2:
        return ImageLinePerspective(
            False, None, predicted_support, float("inf"),
            "Minimaal twee onafhankelijke lange witte 11v11-lijnen vereist.",
        )
    # The atlas already supplies the expected vanishing point.  Intersecting
    # nearly parallel Hough fragments is ill-conditioned: sub-pixel angle
    # noise can move their intersection by thousands of pixels.  Prefer a
    # direct, independent visual confirmation of the predicted direction.
    independent = _independent_line_support(
        predicted_support, minimum_offset_pixels=0.012 * float(np.hypot(*frame_size))
    )
    if len(independent) >= 2:
        residuals = np.asarray(
            [_residual_to_point(item, predicted) for item in independent],
            dtype=np.float64,
        )
        return ImageLinePerspective(
            True,
            (float(predicted[0]), float(predicted[1])),
            independent,
            float(np.sqrt(np.mean(np.square(residuals)))),
            "Voorspelde atlasrichting bevestigd door gescheiden lange witte lijnen.",
        )
    if len(predicted_support) >= 2:
        return ImageLinePerspective(
            False, None, predicted_support, float("inf"),
            "Witte lijnfragmenten volgen de richting maar zijn niet ruimtelijk onafhankelijk.",
        )
    hypotheses = []
    width, height = frame_size
    limit = 12.0 * float(np.hypot(width, height))
    for first, second in combinations(eligible, 2):
        point = _intersection(first, second)
        if point is None or np.linalg.norm(point - predicted) > limit:
            continue
        residuals = np.asarray([_residual_to_point(item, point) for item in eligible])
        selected = residuals <= maximum_residual_degrees
        count = int(np.count_nonzero(selected))
        if count < 2:
            continue
        rms = float(np.sqrt(np.mean(np.square(residuals[selected]))))
        support = float(sum(item.length_pixels for item, keep in zip(eligible, selected) if keep))
        hypotheses.append((-count, rms, -support, point, selected))
    if not hypotheses:
        return ImageLinePerspective(False, None, eligible, float("inf"), "Witte lijnen delen geen stabiel verdwijnpunt.")
    hypotheses.sort(key=lambda item: (item[0], item[1], item[2]))
    _count, rms, _support, point, selected = hypotheses[0]
    supporting = tuple(item for item, keep in zip(eligible, selected) if keep)
    return ImageLinePerspective(
        True,
        (float(point[0]), float(point[1])),
        supporting,
        rms,
        "Gezamenlijk verdwijnpunt uit lange witte 11v11-lijnen.",
    )


def _intersection(first: WhiteLineCandidate, second: WhiteLineCandidate) -> np.ndarray | None:
    first_line = np.cross((*first.start, 1.0), (*first.end, 1.0))
    second_line = np.cross((*second.start, 1.0), (*second.end, 1.0))
    point = np.cross(first_line, second_line)
    if abs(float(point[2])) < 1e-8:
        return None
    point = point[:2] / point[2]
    return point if np.all(np.isfinite(point)) else None


def _independent_line_support(
    candidates: tuple[WhiteLineCandidate, ...], *, minimum_offset_pixels: float
) -> tuple[WhiteLineCandidate, ...]:
    if len(candidates) < 2:
        return ()
    best = None
    for first, second in combinations(candidates, 2):
        first_line = np.cross((*first.start, 1.0), (*first.end, 1.0)).astype(np.float64)
        second_line = np.cross((*second.start, 1.0), (*second.end, 1.0)).astype(np.float64)
        first_line /= max(float(np.linalg.norm(first_line[:2])), 1e-12)
        second_line /= max(float(np.linalg.norm(second_line[:2])), 1e-12)
        first_midpoint = 0.5 * (np.asarray(first.start) + np.asarray(first.end))
        second_midpoint = 0.5 * (np.asarray(second.start) + np.asarray(second.end))
        separation = 0.5 * (
            abs(float(first_line @ (*second_midpoint, 1.0)))
            + abs(float(second_line @ (*first_midpoint, 1.0)))
        )
        score = separation + 0.001 * (first.length_pixels + second.length_pixels)
        if separation >= minimum_offset_pixels and (best is None or score > best[0]):
            best = (score, first, second)
    return () if best is None else (best[1], best[2])


def _residual_to_point(candidate: WhiteLineCandidate, point: np.ndarray) -> float:
    start = np.asarray(candidate.start, dtype=np.float64)
    end = np.asarray(candidate.end, dtype=np.float64)
    midpoint = (start + end) / 2.0
    direction = end - start
    toward = point - midpoint
    denominator = float(np.linalg.norm(direction) * np.linalg.norm(toward))
    if denominator < 1e-9:
        return 90.0
    cosine = np.clip(abs(float(direction @ toward)) / denominator, 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))
