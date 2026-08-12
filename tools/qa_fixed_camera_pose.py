from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.camera_anchor_bank_3d import load_camera_anchor_bank
from football_ai.calibration.fixed_camera_pose import (
    FixedCameraLineConstraint,
    estimate_fixed_camera_poses,
)
from football_ai.calibration.manual_homography_refinement import (
    refine_ground_homography_with_lines,
    refine_ground_homography_with_vanishing_points,
)
from football_ai.calibration.manual_perspective_reference import (
    PerspectiveDirection,
    load_manual_perspective_reference,
)
from football_ai.calibration.reference_3d import create_field_reference_3d
from football_ai.calibration.reference_observation import CameraViewObservations


def main() -> None:
    parser = argparse.ArgumentParser(description="QA voor één vaste camera met pan, tilt en zoom.")
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument(
        "--points-only",
        action="store_true",
        help="Diagnoseer het vaste cameramodel zonder handmatige lijnrichtingen.",
    )
    args = parser.parse_args()

    video_path = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video_path.stem}_{args.format}"
    profile = create_detection_profile(args.format)
    reference = create_field_reference_3d(profile)
    bank = load_camera_anchor_bank(output_dir / f"{prefix}_camera_anchors_3d.json")
    manual = load_manual_perspective_reference(
        output_dir / f"{prefix}_manual_perspective_reference.json"
    )
    ordered_labels = ("left_goal", "right_goal")
    views = []
    manual_views = []
    anchors = []
    for label, goal_id in zip(ordered_labels, ("A", "B")):
        data = json.loads((output_dir / f"{prefix}_view_{goal_id}_3d.json").read_text())
        views.append(CameraViewObservations.from_dict(data["view"]))
        manual_views.append(next(item for item in manual.views if item.label == label))
        anchors.append(next(item for item in bank.anchors if item.goal_id == goal_id))

    left_constraints_by_axis = []
    left_ground = views[0].ground_observations(reference)
    left_points = np.asarray(
        [reference.landmark(item.landmark_id).point.as_tuple()[:2] for item in left_ground]
    )
    left_image = np.asarray([item.image_point for item in left_ground])
    left_refinement = refine_ground_homography_with_lines(
        anchors[0].projection.ground_homography(),
        left_points,
        left_image,
        tuple(item.equation() for item in manual_views[0].lines),
    )
    left_axes = left_refinement.line_axis_assignment

    right_ground = views[1].ground_observations(reference)
    right_points = np.asarray(
        [reference.landmark(item.landmark_id).point.as_tuple()[:2] for item in right_ground]
    )
    right_image = np.asarray([item.image_point for item in right_ground])
    right_refinement = refine_ground_homography_with_lines(
        anchors[1].projection.ground_homography(),
        right_points,
        right_image,
        tuple(item.equation() for item in manual_views[1].lines),
    )
    right_axes = list(right_refinement.line_axis_assignment)

    frame_size = _frame_size(video_path)
    regulatory_pitch_dimension_bounds = (
        (profile.minimum_pitch_length_m, profile.maximum_pitch_length_m),
        (profile.minimum_pitch_width_m, profile.maximum_pitch_width_m),
    )
    pitch_dimension_bounds = profile.soft_pitch_dimension_bounds
    estimate_dimensions = not profile.dimensions_are_exact
    if args.points_only:
        active_constraints = ()
        result = estimate_fixed_camera_poses(
            reference,
            tuple(views),
            frame_size,
            pitch_dimension_bounds=pitch_dimension_bounds if estimate_dimensions else None,
            pitch_dimension_prior_weight=4.0 if estimate_dimensions else 0.0,
        )
        selected_axis_swaps = None
    else:
        candidates = []
        for swap_left in (False, True):
            for swap_right in (False, True):
                constraints = tuple(
                    FixedCameraLineConstraint(
                        0,
                        (1 - axis) if swap_left else axis,
                        line.equation(),
                        0.2,
                    )
                    for line, axis in zip(manual_views[0].lines, left_axes)
                ) + tuple(
                    FixedCameraLineConstraint(
                        1,
                        (1 - axis) if swap_right else axis,
                        line.equation(),
                        0.2,
                    )
                    for line, axis in zip(manual_views[1].lines, right_axes)
                )
                estimate = estimate_fixed_camera_poses(
                    reference,
                    tuple(views),
                    frame_size,
                    constraints,
                    pitch_dimension_bounds=pitch_dimension_bounds if estimate_dimensions else None,
                    pitch_dimension_prior_weight=4.0 if estimate_dimensions else 0.0,
                )
                candidates.append((estimate.rms_error_px, swap_left, swap_right, constraints, estimate))
        _score, swap_left, swap_right, active_constraints, result = min(
            candidates, key=lambda item: item[0]
        )
        selected_axis_swaps = {"left_goal": swap_left, "right_goal": swap_right}
    report = {
        "schema_version": 1,
        "video_name": video_path.name,
        "match_format": args.format,
        "pitch_dimensions": {
            "estimated_length_m": result.pitch_length_m,
            "estimated_width_m": result.pitch_width_m,
            "allowed_length_m": list(pitch_dimension_bounds[0]),
            "allowed_width_m": list(pitch_dimension_bounds[1]),
            "knvb_length_m": list(regulatory_pitch_dimension_bounds[0]),
            "knvb_width_m": list(regulatory_pitch_dimension_bounds[1]),
            "informal_layout_tolerance_m": profile.boundary_layout_tolerance_m,
            "exact": profile.dimensions_are_exact,
        },
        "camera_center_m": result.camera_center.tolist(),
        "rms_error_px": result.rms_error_px,
        "maximum_error_px": result.maximum_error_px,
        "line_constraints_used": len(active_constraints),
        "selected_axis_swaps": selected_axis_swaps,
        "physically_valid": bool(
            0.5 <= result.camera_center[2] <= 30.0
            and result.rms_error_px <= 15.0
            and result.maximum_error_px <= 30.0
        ),
        "views": [
            {
                "frame_number": item.frame_number,
                "focal_length_px": item.focal_length_px,
                "rotation_vector": item.rotation_vector.tolist(),
                "projection_matrix": item.projection.matrix.tolist(),
                "ground_homography": item.projection.ground_homography().tolist(),
                "rms_error_px": item.rms_error_px,
                "maximum_error_px": item.maximum_error_px,
            }
            for item in result.views
        ],
    }
    report_path = output_dir / f"{prefix}_fixed_camera_pose_qa.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    preview = _preview(video_path, result.views, result.pitch_length_m, result.pitch_width_m)
    preview_path = output_dir / f"{prefix}_fixed_camera_pose_qa.jpg"
    cv2.imwrite(str(preview_path), preview)
    status = "GELDIG" if report["physically_valid"] else "AFGEKEURD"
    print(
        f"Vaste camera: {status} | positie "
        f"({result.camera_center[0]:.2f}, {result.camera_center[1]:.2f}, {result.camera_center[2]:.2f})m | "
        f"RMS {result.rms_error_px:.2f}px | max {result.maximum_error_px:.2f}px"
    )
    print(
        f"Veldmaat: {result.pitch_length_m:.2f} x {result.pitch_width_m:.2f}m | "
        f"praktische zoekruimte {pitch_dimension_bounds[0][0]:g}-{pitch_dimension_bounds[0][1]:g} x "
        f"{pitch_dimension_bounds[1][0]:g}-{pitch_dimension_bounds[1][1]:g}m"
    )
    for item in result.views:
        print(
            f"Frame {item.frame_number}: focal {item.focal_length_px:.0f}px | "
            f"RMS {item.rms_error_px:.2f}px | max {item.maximum_error_px:.2f}px"
        )
    print(f"QA-preview: {preview_path}")
    print(f"QA-rapport: {report_path}")


def _frame_size(video_path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return size


def _preview(video_path: Path, poses, length: float, width: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    panels = []
    corners = np.asarray(((0.0, 0.0, 1.0), (length, 0.0, 1.0), (length, width, 1.0), (0.0, width, 1.0)))
    for pose in poses:
        capture.set(cv2.CAP_PROP_POS_FRAMES, pose.frame_number)
        success, frame = capture.read()
        if not success:
            continue
        homography = pose.projection.ground_homography()
        projected = (homography @ corners.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        cv2.polylines(frame, [np.rint(projected).astype(np.int32)], True, (0, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"frame {pose.frame_number} | RMS {pose.rms_error_px:.1f}px",
            (18, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(frame)
    capture.release()
    if not panels:
        raise RuntimeError("Geen previewframes gelezen.")
    height = min(item.shape[0] for item in panels)
    resized = [cv2.resize(item, (round(item.shape[1] * height / item.shape[0]), height)) for item in panels]
    return np.hstack(resized)


if __name__ == "__main__":
    main()
