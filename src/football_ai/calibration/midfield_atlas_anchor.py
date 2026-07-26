from __future__ import annotations

import cv2
import numpy as np

from football_ai.calibration.local_field_atlas import LocalFieldPatch


def create_midfield_patch(
    anchor_frame: int,
    ground_to_reference: np.ndarray,
    midfield_to_reference: np.ndarray,
    pitch_length_m: float,
    pitch_width_m: float,
    graph_rms_px: float,
    maximum_graph_error_px: float,
) -> LocalFieldPatch:
    """Create a metric center patch transported through a validated frame graph."""
    reference = np.asarray(ground_to_reference, dtype=np.float64)
    motion = np.asarray(midfield_to_reference, dtype=np.float64)
    if reference.shape != (3, 3) or motion.shape != (3, 3):
        raise ValueError("Middenanker vereist twee geldige 3x3-transformaties.")
    ground_to_midfield = np.linalg.inv(motion) @ reference
    ground_to_midfield /= ground_to_midfield[2, 2]
    support = np.asarray(
        (
            (0.25 * pitch_length_m, 0.0),
            (0.75 * pitch_length_m, 0.0),
            (0.75 * pitch_length_m, pitch_width_m),
            (0.25 * pitch_length_m, pitch_width_m),
        ),
        dtype=np.float64,
    )
    projected = _project(support, ground_to_midfield)
    if (
        not np.all(np.isfinite(projected))
        or not cv2.isContourConvex(projected.astype(np.float32).reshape(-1, 1, 2))
    ):
        raise ValueError("De getransporteerde middenzone vormt geen convex grondvlak.")
    if graph_rms_px > 8.0 or maximum_graph_error_px > 24.0:
        raise ValueError(
            "De framegraph is te onnauwkeurig voor een zelfstandig middenanker: "
            f"RMS {graph_rms_px:.1f}px, max {maximum_graph_error_px:.1f}px."
        )
    confidence = float(
        np.clip(np.exp(-graph_rms_px / 8.0) * np.exp(-maximum_graph_error_px / 30.0), 0.10, 0.90)
    )
    return LocalFieldPatch(
        "midfield",
        anchor_frame,
        ground_to_midfield,
        tuple(tuple(map(float, item)) for item in support),
        confidence,
        "framegraph vanaf doel-b + handmatige 11v11-middenlijn",
        (),
        ("sideline_rear", "sideline_front"),
    )


def create_bridged_midfield_patch(
    anchor_frame: int,
    ground_to_midfield_from_goal_a: np.ndarray,
    ground_to_midfield_from_goal_b: np.ndarray,
    pitch_length_m: float,
    pitch_width_m: float,
    graph_rms_px: float,
    maximum_graph_error_px: float,
) -> LocalFieldPatch:
    """Solve one center homography from the independently transported end lines."""
    from_a = np.asarray(ground_to_midfield_from_goal_a, dtype=np.float64)
    from_b = np.asarray(ground_to_midfield_from_goal_b, dtype=np.float64)
    if from_a.shape != (3, 3) or from_b.shape != (3, 3):
        raise ValueError("Middenbrug vereist twee geldige 3x3-projecties.")
    ground_corners = np.asarray(
        (
            (0.0, 0.0),
            (pitch_length_m, 0.0),
            (pitch_length_m, pitch_width_m),
            (0.0, pitch_width_m),
        ),
        dtype=np.float64,
    )
    image_corners = np.vstack(
        (
            _project(ground_corners[[0]], from_a),
            _project(ground_corners[[1, 2]], from_b),
            _project(ground_corners[[3]], from_a),
        )
    )
    if (
        not np.all(np.isfinite(image_corners))
        or not cv2.isContourConvex(image_corners.astype(np.float32).reshape(-1, 1, 2))
    ):
        raise ValueError("De vanuit beide doelen getransporteerde hoeken vormen geen convex veld.")
    ground_to_midfield = cv2.getPerspectiveTransform(
        ground_corners.astype(np.float32), image_corners.astype(np.float32)
    ).astype(np.float64)
    ground_to_midfield /= ground_to_midfield[2, 2]
    if graph_rms_px > 8.0 or maximum_graph_error_px > 24.0:
        raise ValueError(
            "De framegraph is te onnauwkeurig voor een gekoppeld middenanker: "
            f"RMS {graph_rms_px:.1f}px, max {maximum_graph_error_px:.1f}px."
        )
    center = np.asarray(((0.5 * pitch_length_m, 0.0), (0.5 * pitch_length_m, pitch_width_m)))
    disagreement = float(
        np.mean(np.linalg.norm(_project(center, from_a) - _project(center, from_b), axis=1))
    )
    confidence = float(
        np.clip(
            np.exp(-graph_rms_px / 8.0)
            * np.exp(-maximum_graph_error_px / 30.0)
            * np.exp(-disagreement / 180.0),
            0.10,
            0.90,
        )
    )
    support = (
        (0.25 * pitch_length_m, 0.0),
        (0.75 * pitch_length_m, 0.0),
        (0.75 * pitch_length_m, pitch_width_m),
        (0.25 * pitch_length_m, pitch_width_m),
    )
    return LocalFieldPatch(
        "midfield",
        anchor_frame,
        ground_to_midfield,
        support,
        confidence,
        "vier veldhoeken gekoppeld vanuit doel-a en doel-b",
        (),
        ("sideline_rear", "sideline_front"),
    )


def create_positioned_midfield_patch(
    anchor_frame: int,
    ground_to_midfield_from_goal_a: np.ndarray,
    ground_to_midfield_from_goal_b: np.ndarray,
    longitudinal_reference_line: np.ndarray,
    rear_sideline_point: tuple[float, float] | None,
    front_sideline_point: tuple[float, float] | None,
    pitch_length_m: float,
    pitch_width_m: float,
    graph_rms_px: float,
    maximum_graph_error_px: float,
    patch_id: str = "midfield",
) -> LocalFieldPatch:
    """Intersect transported end lines with two manually positioned sidelines."""
    from_a = np.asarray(ground_to_midfield_from_goal_a, dtype=np.float64)
    from_b = np.asarray(ground_to_midfield_from_goal_b, dtype=np.float64)
    reference_line = np.asarray(longitudinal_reference_line, dtype=np.float64)
    reference_line /= np.linalg.norm(reference_line[:2])
    vanishing_candidates = []
    for matrix in (from_a, from_b):
        point = matrix[:, 0]
        if abs(float(point[2])) > 1e-9:
            vanishing_candidates.append(point[:2] / point[2])
    if not vanishing_candidates:
        raise ValueError("De getransporteerde doelvlakken leveren geen eindig zijlijnverdwijnpunt.")
    vanishing = np.mean(vanishing_candidates, axis=0)
    vanishing -= float(reference_line @ np.asarray((*vanishing, 1.0))) * reference_line[:2]
    vanishing_h = np.asarray((*vanishing, 1.0), dtype=np.float64)
    if rear_sideline_point is None and front_sideline_point is None:
        raise ValueError("Minimaal één zichtbare 8v8-zijlijnpositie is vereist.")
    center_x = 0.5 * pitch_length_m
    rear_predictions = tuple(
        _project(np.asarray(((center_x, 0.0),)), matrix)[0] for matrix in (from_b, from_a)
    )
    front_predictions = tuple(
        _project(np.asarray(((center_x, pitch_width_m),)), matrix)[0]
        for matrix in (from_b, from_a)
    )
    rear_candidates = (
        (np.asarray(rear_sideline_point),)
        if rear_sideline_point is not None
        else (np.mean(rear_predictions, axis=0), *rear_predictions)
    )
    front_candidates = (
        (np.asarray(front_sideline_point),)
        if front_sideline_point is not None
        else (np.mean(front_predictions, axis=0), *front_predictions)
    )
    end_a_points = _project(np.asarray(((0.0, 0.0), (0.0, pitch_width_m))), from_a)
    end_b_points = _project(
        np.asarray(((pitch_length_m, 0.0), (pitch_length_m, pitch_width_m))), from_b
    )
    end_a_line = np.cross(
        np.asarray((*end_a_points[0], 1.0)), np.asarray((*end_a_points[1], 1.0))
    )
    end_b_line = np.cross(
        np.asarray((*end_b_points[0], 1.0)), np.asarray((*end_b_points[1], 1.0))
    )
    image_corners = None
    last_candidate = None
    for rear_position in rear_candidates:
        rear_line = np.cross(np.asarray((*rear_position, 1.0)), vanishing_h)
        for front_position in front_candidates:
            front_line = np.cross(np.asarray((*front_position, 1.0)), vanishing_h)
            candidate = np.asarray(
                (
                    _intersection(end_a_line, rear_line),
                    _intersection(end_b_line, rear_line),
                    _intersection(end_b_line, front_line),
                    _intersection(end_a_line, front_line),
                ),
                dtype=np.float64,
            )
            last_candidate = candidate
            if (
                np.all(np.isfinite(candidate))
                and cv2.isContourConvex(candidate.astype(np.float32).reshape(-1, 1, 2))
                and abs(float(cv2.contourArea(candidate.astype(np.float32)))) > 1000.0
            ):
                image_corners = candidate
                break
        if image_corners is not None:
            break
    ground_corners = np.asarray(
        ((0.0, 0.0), (pitch_length_m, 0.0), (pitch_length_m, pitch_width_m), (0.0, pitch_width_m)),
        dtype=np.float64,
    )
    if (
        image_corners is None
        or not np.all(np.isfinite(image_corners))
    ):
        detail = "onbekend" if last_candidate is None else np.array2string(
            last_candidate, precision=1, suppress_small=True
        )
        raise ValueError(
            "De twee zijlijnposities en getransporteerde achterlijnen vormen geen convex veld. "
            f"Getransporteerde punten: achter={np.asarray(rear_sideline_point)}, "
            f"voor={np.asarray(front_sideline_point)}; hoeken={detail}"
        )
    matrix = cv2.getPerspectiveTransform(
        ground_corners.astype(np.float32), image_corners.astype(np.float32)
    ).astype(np.float64)
    matrix /= matrix[2, 2]
    confidence = float(
        np.clip(
            np.exp(-graph_rms_px / 8.0) * np.exp(-maximum_graph_error_px / 30.0),
            0.10,
            0.90,
        )
    )
    support = (
        (0.25 * pitch_length_m, 0.0),
        (0.75 * pitch_length_m, 0.0),
        (0.75 * pitch_length_m, pitch_width_m),
        (0.25 * pitch_length_m, pitch_width_m),
    )
    verified = []
    inferred = []
    if rear_sideline_point is not None:
        verified.append("sideline_rear")
    else:
        inferred.append("sideline_rear")
    if front_sideline_point is not None:
        verified.append("sideline_front")
    else:
        inferred.append("sideline_front")
    return LocalFieldPatch(
        patch_id, anchor_frame, matrix, support, confidence,
        "achterlijnen uit doelvlakken + zichtbare/geïnterpoleerde zijlijnposities",
        tuple(verified), tuple(inferred),
    )


def midfield_direction_error_px(
    patch: LocalFieldPatch,
    image_line: np.ndarray,
) -> float:
    line = np.asarray(image_line, dtype=np.float64)
    line /= np.linalg.norm(line[:2])
    vanishing = patch.ground_to_anchor[:, 0]
    if abs(float(vanishing[2])) < 1e-9:
        return float("inf")
    return abs(float(line @ (vanishing / vanishing[2])))


def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (matrix @ homogeneous.T).T
    if np.any(np.abs(projected[:, 2]) < 1e-9):
        raise ValueError("Middenanker projecteert naar oneindig.")
    return projected[:, :2] / projected[:, 2:3]


def _intersection(first_line: np.ndarray, second_line: np.ndarray) -> tuple[float, float]:
    point = np.cross(first_line, second_line)
    if abs(float(point[2])) < 1e-9:
        raise ValueError("Twee vereiste veldlijnen snijden niet in een eindig hoekpunt.")
    point /= point[2]
    return float(point[0]), float(point[1])
