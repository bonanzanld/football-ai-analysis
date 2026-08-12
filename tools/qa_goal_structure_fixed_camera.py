from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.bootstrap.goal_seed import load_goal_seeds
from football_ai.calibration.fixed_camera_pose import (
    FixedCameraLineConstraint,
    FixedCameraPointConstraint,
    estimate_fixed_camera_poses,
)
from football_ai.calibration.global_frame_graph import estimate_frame_edge, estimate_ground_frame_edge
from football_ai.calibration.goal_structure_observation import load_goal_structure_observations
from football_ai.calibration.lens_geometry import LensIntrinsics
from football_ai.calibration.manual_perspective_reference import (
    PerspectiveDirection,
    load_manual_perspective_reference,
)
from football_ai.calibration.reference_3d import create_field_reference_3d
from football_ai.calibration.reference_observation import CameraViewObservations, ReferenceObservation2D


def main() -> None:
    parser = argparse.ArgumentParser(description="QA voor één camera uit twee doelstructuren.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    lens_data = json.loads((output_dir / f"{prefix}_lens_geometry_qa.json").read_text())
    if not lens_data.get("stable", False):
        raise RuntimeError("Het lensprofiel is niet stabiel goedgekeurd.")
    lens = LensIntrinsics(
        tuple(lens_data["frame_size"]),
        float(lens_data["focal_length_px"]),
        tuple(lens_data["principal_point"]),
        tuple(lens_data["radial_distortion"]),
    )
    structures = load_goal_structure_observations(output_dir / f"{prefix}_goal_structure_lines.json")
    seeds = {item.goal_id: item for item in load_goal_seeds(output_dir / f"{prefix}_goal_seeds.json")}
    profile = create_detection_profile(args.format)
    reference = create_field_reference_3d(profile)
    prepared = []
    for structure in structures:
        corners = structure.corners()
        raw = np.asarray([corners[name] for name in ("far_bottom", "far_top", "near_top", "near_bottom")])
        corrected = lens.undistort_points(raw)
        seed = seeds[structure.goal_id]
        prepared.append((structure, seed, corrected))
    perspective_path = output_dir / f"{prefix}_manual_perspective_reference.json"
    line_constraints, perspective_diagnostics = _prepare_perspective_constraints(
        video, structures, lens, perspective_path
    )
    candidates = []
    for swap_a in (False, True):
        for swap_b in (False, True):
            swaps = {"A": swap_a, "B": swap_b}
            views = []
            point_constraints = []
            for structure, seed, corrected in prepared:
                goal = structure.goal_id.lower()
                landmark_ids = (
                    f"goal_{goal}_rear_bottom", f"goal_{goal}_rear_top",
                    f"goal_{goal}_front_top", f"goal_{goal}_front_bottom",
                )
                image_points = corrected
                if swaps[structure.goal_id]:
                    image_points = corrected[[3, 2, 1, 0]]
                observations = tuple(
                    ReferenceObservation2D(name, tuple(point))
                    for name, point in zip(landmark_ids, image_points)
                )
                views.append(CameraViewObservations(structure.frame_number, seed.camera_state, observations))
                if seed.front_corner is not None:
                    landmark = reference.landmark(f"corner_{goal}_front").point
                    image_point = lens.undistort_points(
                        np.asarray((seed.front_corner,), dtype=np.float64)
                    )[0]
                    point_constraints.append(
                        FixedCameraPointConstraint(
                            len(views) - 1,
                            landmark.as_tuple(),
                            tuple(image_point),
                            1.0,
                        )
                    )
            try:
                estimate = estimate_fixed_camera_poses(
                    reference,
                    tuple(views),
                    lens.frame_size,
                    line_constraints=line_constraints,
                    point_constraints=tuple(point_constraints),
                    focal_length_prior_px=lens.focal_length_px,
                    focal_prior_weight=80.0,
                    shared_focal_weight=120.0,
                    camera_height_prior_m=3.75,
                    camera_height_weight=80.0,
                    camera_center_prior_xy=(
                        profile.pitch_length_m / 2.0,
                        profile.pitch_width_m + 4.0,
                    ),
                    camera_center_weight=4.0,
                    pitch_dimension_bounds=profile.soft_pitch_dimension_bounds,
                    pitch_dimension_prior_weight=16.0,
                    camera_outside_clearance_m=0.5,
                    camera_outside_weight=100.0,
                    camera_x_bounds=(0.0, profile.soft_pitch_dimension_bounds[0][1]),
                    estimate_principal_point=True,
                    principal_point_prior_weight=40.0,
                    principal_point_max_shift_ratio=0.2,
                )
            except (ValueError, cv2.error):
                continue
            height_penalty = 20.0 * max(0.0, 2.0 - estimate.camera_center[2])
            corner_errors = []
            for (structure, seed, _corrected), pose in zip(prepared, estimate.views):
                if seed.front_corner is None:
                    continue
                observed = lens.undistort_points(
                    np.asarray((seed.front_corner,), dtype=np.float64)
                )[0]
                corner_x = 0.0 if structure.goal_id == "A" else estimate.pitch_length_m
                predicted = np.asarray(
                    pose.projection.project((corner_x, estimate.pitch_width_m, 0.0)),
                    dtype=np.float64,
                )
                corner_errors.append(float(np.linalg.norm(predicted - observed)))
            holdout_rms = (
                float(np.sqrt(np.mean(np.square(corner_errors))))
                if corner_errors else float("inf")
            )
            camera_side_penalty = (
                0.0
                if estimate.camera_center[1] > estimate.pitch_width_m
                else 1000.0 + 20.0 * (estimate.pitch_width_m - estimate.camera_center[1])
            )
            geometry_valid, geometry_penalty = _field_geometry_check(estimate)
            candidates.append(
                (
                    estimate.rms_error_px + height_penalty + 0.05 * holdout_rms
                    + camera_side_penalty + geometry_penalty,
                    swaps,
                    estimate,
                    holdout_rms,
                    geometry_valid,
                )
            )
    if not candidates:
        raise RuntimeError("Geen gezamenlijke camerapose gevonden voor de twee doelen.")
    _score, selected_swaps, result, corner_holdout_rms, geometry_valid = min(
        candidates, key=lambda item: item[0]
    )
    focal_ratios = tuple(item.focal_length_px / lens.focal_length_px for item in result.views)
    camera_outside_field = bool(result.camera_center[1] > result.pitch_width_m)
    physically_valid = bool(
        2.5 <= result.camera_center[2] <= 6.0
        and camera_outside_field
        and result.rms_error_px <= 12.0
        and result.maximum_error_px <= 25.0
        and all(0.65 <= ratio <= 1.60 for ratio in focal_ratios)
        and max(focal_ratios) / min(focal_ratios) <= 1.25
        and corner_holdout_rms <= 80.0
        and geometry_valid
    )
    report = {
        "schema_version": 1,
        "video_name": video.name,
        "lens_corrected": True,
        "camera_center_m": result.camera_center.tolist(),
        "estimated_pitch_length_m": result.pitch_length_m,
        "estimated_pitch_width_m": result.pitch_width_m,
        "nominal_pitch_length_m": profile.pitch_length_m,
        "nominal_pitch_width_m": profile.pitch_width_m,
        "rms_error_px": result.rms_error_px,
        "maximum_error_px": result.maximum_error_px,
        "physically_valid": physically_valid,
        "focal_ratios_to_lens_prior": list(focal_ratios),
        "camera_outside_playable_field": camera_outside_field,
        "field_corner_holdout_rms_px": corner_holdout_rms,
        "field_geometry_visible_in_front_of_camera": geometry_valid,
        "selected_post_swaps": selected_swaps,
        "perspective_constraints": perspective_diagnostics,
        "field_corners_used_as_fit_points": False,
        "views": [
            {
                "frame_number": item.frame_number,
                "focal_length_px": item.focal_length_px,
                "rms_error_px": item.rms_error_px,
                "maximum_error_px": item.maximum_error_px,
                "principal_point": list(item.principal_point),
                "projection_matrix_undistorted": item.projection.matrix.tolist(),
            }
            for item in result.views
        ],
    }
    report_path = output_dir / f"{prefix}_goal_structure_fixed_camera_qa.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    preview_path = output_dir / f"{prefix}_goal_structure_fixed_camera_qa.jpg"
    _write_preview(video, result, lens, result.pitch_length_m, result.pitch_width_m, preview_path)
    print(
        f"Vaste-camera-QA: {'PASS' if physically_valid else 'FAIL'} | "
        f"positie ({result.camera_center[0]:.2f}, {result.camera_center[1]:.2f}, {result.camera_center[2]:.2f})m | "
        f"RMS {result.rms_error_px:.2f}px | max {result.maximum_error_px:.2f}px"
    )
    print(
        f"Geschat speelveld: {result.pitch_length_m:.2f} x {result.pitch_width_m:.2f}m "
        f"(voorkeur {profile.pitch_length_m:.1f} x {profile.pitch_width_m:.1f}m)"
    )
    print(f"Paaloriëntatie omgewisseld: A={selected_swaps['A']} | B={selected_swaps['B']}")
    print(f"Hoedjeshoeken hold-out RMS: {corner_holdout_rms:.2f}px")
    print(
        f"Witte perspectiefsteun: {len(line_constraints)} lijnen | "
        + " | ".join(perspective_diagnostics)
    )
    for item in result.views:
        print(f"Frame {item.frame_number}: focal {item.focal_length_px:.0f}px | RMS {item.rms_error_px:.2f}px | max {item.maximum_error_px:.2f}px")
    print(f"QA-preview: {preview_path}")
    print(f"QA-rapport: {report_path}")


def _write_preview(video, result, lens, length, width, path):
    capture = cv2.VideoCapture(str(video))
    panels = []
    ground = np.asarray(((0.0, 0.0, 1.0), (length, 0.0, 1.0), (length, width, 1.0), (0.0, width, 1.0)))
    for pose in result.views:
        capture.set(cv2.CAP_PROP_POS_FRAMES, pose.frame_number)
        ok, frame = capture.read()
        if not ok:
            continue
        corrected = cv2.undistort(frame, lens.camera_matrix, lens.distortion_coefficients)
        h = pose.projection.ground_homography()
        points = (h @ ground.T).T
        points = points[:, :2] / points[:, 2:3]
        cv2.polylines(corrected, [np.rint(points).astype(np.int32)], True, (0, 255, 255), 7, cv2.LINE_AA)
        panels.append(cv2.resize(corrected, (960, 540)))
    capture.release()
    if not panels or not cv2.imwrite(str(path), np.hstack(panels)):
        raise RuntimeError("Vaste-camera-preview kon niet worden opgeslagen.")


def _field_geometry_check(result) -> tuple[bool, float]:
    corners = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (result.pitch_length_m, 0.0, 0.0),
            (result.pitch_length_m, result.pitch_width_m, 0.0),
            (0.0, result.pitch_width_m, 0.0),
        ),
        dtype=np.float64,
    )
    penalty = 0.0
    valid = True
    for pose in result.views:
        rotation, _ = cv2.Rodrigues(pose.rotation_vector)
        depths = (rotation @ (corners - result.camera_center).T)[2]
        behind = int(np.count_nonzero(depths <= 0.1))
        if behind:
            valid = False
            penalty += 10000.0 * behind
        try:
            projected = np.asarray(
                [pose.projection.project(tuple(point)) for point in corners], dtype=np.float32
            )
            if not cv2.isContourConvex(np.rint(projected).astype(np.int32)):
                valid = False
                penalty += 10000.0
        except (ValueError, cv2.error):
            valid = False
            penalty += 50000.0
    return valid, penalty


def _prepare_perspective_constraints(video, structures, lens, path):
    if not path.exists():
        return (), ("geen handmatige perspectiefreferentie",)
    reference = load_manual_perspective_reference(path)
    views_by_label = {item.label: item for item in reference.views}
    structure_by_goal = {item.goal_id: item for item in structures}
    capture = cv2.VideoCapture(str(video))
    constraints = []
    diagnostics = []
    try:
        for view_index, (goal_id, label) in enumerate((("A", "left_goal"), ("B", "right_goal"))):
            perspective_view = views_by_label[label]
            structure = structure_by_goal[goal_id]
            transform = np.eye(3, dtype=np.float64)
            if perspective_view.frame_number != structure.frame_number:
                source = _read_frame(capture, perspective_view.frame_number)
                target = _read_frame(capture, structure.frame_number)
                try:
                    edge = estimate_ground_frame_edge(label, f"goal-{goal_id}", source, target)
                    registration_kind = "grond"
                except ValueError:
                    edge = estimate_frame_edge(label, f"goal-{goal_id}", source, target)
                    registration_kind = "volledig beeld"
                transform = edge.source_to_target
                diagnostics.append(
                    f"{goal_id}: framekoppeling {perspective_view.frame_number}->{structure.frame_number}, "
                    f"{edge.inliers} inliers/{edge.inlier_ratio:.0%} ({registration_kind})"
                )
            else:
                diagnostics.append(f"{goal_id}: zelfde frame {structure.frame_number}")
            for line in perspective_view.lines:
                if line.direction is PerspectiveDirection.UNKNOWN:
                    continue
                raw = np.asarray(line.points, dtype=np.float64)
                mapped = cv2.perspectiveTransform(raw.reshape(1, -1, 2), transform).reshape(-1, 2)
                corrected = lens.undistort_points(mapped)
                equation = _fit_image_line(corrected)
                world_axis = 0 if line.direction is PerspectiveDirection.BETWEEN_GOALS else 1
                # Vanishing points can lie thousands of pixels outside the image.
                # Keep these as directional evidence, not as dominant metric points.
                constraints.append(FixedCameraLineConstraint(view_index, world_axis, equation, 0.01))
    finally:
        capture.release()
    return tuple(constraints), tuple(diagnostics)


def _read_frame(capture, frame_number):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Videoframe {frame_number} kon niet worden gelezen.")
    return frame


def _fit_image_line(points):
    center = np.mean(points, axis=0)
    _u, _s, vh = np.linalg.svd(points - center)
    direction = vh[0]
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    equation = np.asarray((normal[0], normal[1], -float(normal @ center)))
    return equation / np.linalg.norm(equation[:2])


if __name__ == "__main__":
    main()
