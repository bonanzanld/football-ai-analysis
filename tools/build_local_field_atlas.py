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
from football_ai.calibration.bootstrap.goal_seed import estimate_backline_endpoints, load_goal_seeds
from football_ai.calibration.goal_plane_camera import estimate_camera_from_goal_plane
from football_ai.calibration.goal_structure_observation import load_goal_structure_observations
from football_ai.calibration.lens_geometry import LensIntrinsics
from football_ai.calibration.local_field_atlas import (
    LocalFieldAtlas,
    LocalFieldPatch,
    anchor_patch_to_measured_endline,
    align_patch_to_front_sideline,
    save_local_field_atlas,
)
from football_ai.calibration.global_frame_graph import estimate_frame_edge, estimate_ground_frame_edge
from football_ai.calibration.manual_perspective_reference import (
    PerspectiveDirection,
    load_manual_perspective_reference,
)
from football_ai.calibration.manual_midfield_line import load_manual_midfield_line
from football_ai.calibration.manual_parallel_lines import load_manual_parallel_lines
from football_ai.calibration.reference_3d import create_field_reference_3d
from football_ai.calibration.reference_observation import CameraViewObservations, ReferenceObservation2D


def main() -> None:
    parser = argparse.ArgumentParser(description="Bouw lokale, overlappende speelveldvlakken.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    profile = create_detection_profile(args.format)
    reference = create_field_reference_3d(profile)
    lens_data = json.loads((output / f"{prefix}_lens_geometry_qa.json").read_text())
    lens = LensIntrinsics(
        tuple(lens_data["frame_size"]), float(lens_data["focal_length_px"]),
        tuple(lens_data["principal_point"]), tuple(lens_data["radial_distortion"]),
    )
    structures = load_goal_structure_observations(output / f"{prefix}_goal_structure_lines.json")
    seeds = {item.goal_id: item for item in load_goal_seeds(output / f"{prefix}_goal_seeds.json")}
    perspective = load_manual_perspective_reference(
        output / f"{prefix}_manual_perspective_reference.json"
    )
    midfield = load_manual_midfield_line(output / f"{prefix}_manual_midfield_line.json")
    parallel_reference = load_manual_parallel_lines(
        output / f"{prefix}_manual_parallel_lines.json"
    )
    if midfield.video_name != video.name:
        raise RuntimeError("De handmatige 11v11-middenlijn hoort bij een andere video.")
    if parallel_reference.video_name != video.name:
        raise RuntimeError("De parallelle 11v11-lijnen horen bij een andere video.")
    vanishing_points, parallel_diagnostics = _goal_vanishing_points(
        video, structures, perspective, lens, seeds, parallel_reference
    )
    overlap = 0.25 * profile.pitch_length_m
    patches = []
    diagnostics = []
    for structure in structures:
        seed = seeds[structure.goal_id]
        corners = structure.corners()
        raw = np.asarray([corners[name] for name in ("far_bottom", "far_top", "near_top", "near_bottom")])
        corrected = lens.undistort_points(raw)
        goal = structure.goal_id.lower()
        names = [
            f"goal_{goal}_rear_bottom", f"goal_{goal}_rear_top",
            f"goal_{goal}_front_top", f"goal_{goal}_front_bottom",
        ]
        candidates = []
        for swapped in (False, True):
            image_points = corrected[[3, 2, 1, 0]] if swapped else corrected
            if seed.front_corner is None:
                raise RuntimeError(f"Doel {structure.goal_id} mist de gemeten hoek voor zijn lokale vlak.")
            image_points = np.vstack(
                (image_points, lens.undistort_points(np.asarray((seed.front_corner,), dtype=np.float64)))
            )
            local_names = (*names, f"corner_{goal}_front")
            view = CameraViewObservations(
                structure.frame_number, seed.camera_state,
                tuple(ReferenceObservation2D(name, tuple(point)) for name, point in zip(local_names, image_points)),
            )
            try:
                estimate = estimate_camera_from_goal_plane(
                    reference, view, lens.frame_size,
                    ground_direction_vanishing_point=vanishing_points[structure.goal_id],
                )
            except (ValueError, cv2.error):
                continue
            corner_error = float("inf")
            if seed.front_corner is not None:
                observed = lens.undistort_points(np.asarray((seed.front_corner,), dtype=np.float64))[0]
                predicted = np.asarray(
                    estimate.projection.project(reference.landmark(f"corner_{goal}_front").point)
                )
                corner_error = float(np.linalg.norm(predicted - observed))
            candidates.append((estimate.rms_error_px + 0.04 * corner_error, swapped, estimate, corner_error))
        if not candidates:
            raise RuntimeError(f"Doel {structure.goal_id} levert geen lokaal cameravlak.")
        _score, swapped, estimate, corner_error = min(candidates, key=lambda item: item[0])
        if structure.goal_id == "A":
            minimum_x, maximum_x = 0.0, profile.pitch_length_m / 2.0 + overlap / 2.0
        else:
            minimum_x, maximum_x = profile.pitch_length_m / 2.0 - overlap / 2.0, profile.pitch_length_m
        support = (
            (minimum_x, 0.0), (maximum_x, 0.0),
            (maximum_x, profile.pitch_width_m), (minimum_x, profile.pitch_width_m),
        )
        confidence = float(np.clip(np.exp(-estimate.rms_error_px / 8.0) * np.exp(-corner_error / 120.0), 0.05, 1.0))
        ground_homography, reflected = _orient_ground_homography(
            estimate.projection.ground_homography(), structure.goal_id, seed, profile, lens
        )
        rear_corner, front_corner = estimate_backline_endpoints(
            seed.first_ground, seed.second_ground, seed.goal_width_m,
            profile.pitch_width_m, seed.rear_corner, seed.front_corner,
        )
        corrected_endpoints = lens.undistort_points(
            np.asarray((rear_corner, front_corner), dtype=np.float64)
        )
        sideline_points = lens.undistort_points(
            np.asarray(seed.front_sideline_observations, dtype=np.float64)
        )
        effective_vanishing = _bind_vanishing_point_to_observed_sideline(
            vanishing_points[structure.goal_id], tuple(corrected_endpoints[1]),
            sideline_points,
        )
        ground_homography, sideline_rms = align_patch_to_front_sideline(
            ground_homography, structure.goal_id, profile.pitch_length_m,
            profile.pitch_width_m, sideline_points,
            direction_vanishing_point=effective_vanishing,
        )
        # Apply hard physical anchors last. A later scale-only refinement can
        # otherwise silently reverse the field-inward direction again.
        ground_homography = anchor_patch_to_measured_endline(
            ground_homography, structure.goal_id, profile.pitch_length_m,
            profile.pitch_width_m, tuple(corrected_endpoints[0]),
            tuple(corrected_endpoints[1]), effective_vanishing,
            front_sideline_points=sideline_points,
        )
        patches.append(
            LocalFieldPatch(
                f"goal-{goal}", structure.frame_number, ground_homography,
                support, confidence, "5x2m-doelvlak + gemeten veldhoek",
                (
                    "end_line_a" if structure.goal_id == "A" else "end_line_b",
                    "sideline_front",
                ),
                ("sideline_rear",),
            )
        )
        diagnostics.append(
            (
                structure.goal_id, swapped, reflected, estimate.rms_error_px,
                corner_error, sideline_rms, confidence,
            )
        )
    atlas = LocalFieldAtlas(
        video.name, args.format, profile.pitch_length_m, profile.pitch_width_m,
        tuple(patches), midfield, parallel_reference,
    )
    atlas_path = output / f"{prefix}_local_field_atlas.json"
    save_local_field_atlas(atlas, atlas_path)
    preview_path = output / f"{prefix}_local_field_atlas_qa.jpg"
    _write_preview(video, atlas, lens, preview_path)
    print(f"Lokale veldatlas opgeslagen: {atlas_path}")
    for goal, swapped, reflected, rms, corner, sideline_rms, confidence in diagnostics:
        print(
            f"Vlak {goal}: doel-RMS {rms:.2f}px | hoekcontrole {corner:.2f}px | "
            f"zijlijn-RMS {sideline_rms:.2f}px | vertrouwen {confidence:.0%} | "
            f"palen omgewisseld={swapped} | vlak gespiegeld={reflected}"
        )
    print(f"Overlapzone: {overlap:.1f}m rond het midden")
    print(
        f"11v11-middenlijnanker: frame {midfield.frame_number} | "
        f"RMS {midfield.rms_error_px:.2f}px | richting = parallel aan 8v8-zijlijnen"
    )
    for goal_id, diagnostic in parallel_diagnostics.items():
        print(
            f"Parallel-crosscheck {goal_id}: 5m+16m leidend | "
            f"verschil met automatische lijnfamilie {diagnostic['difference_px']:.1f}px "
            f"({diagnostic['difference_degrees']:.2f} graden)"
        )
    print(f"QA-preview: {preview_path}")


def _write_preview(video, atlas, lens, path):
    capture = cv2.VideoCapture(str(video))
    panels = []
    colors = ((0, 230, 255), (255, 180, 0))
    try:
        for patch, color in zip(atlas.patches, colors):
            capture.set(cv2.CAP_PROP_POS_FRAMES, patch.anchor_frame)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Ankerframe {patch.anchor_frame} kon niet worden gelezen.")
            corrected = cv2.undistort(frame, lens.camera_matrix, lens.distortion_coefficients)
            evidence = atlas.visible_evidence(
                patch.patch_id, (corrected.shape[1], corrected.shape[0])
            )
            if len(evidence.visible_polygon) >= 3:
                overlay = corrected.copy()
                polygon = np.rint(evidence.visible_polygon).astype(np.int32)
                cv2.fillPoly(overlay, [polygon], color)
                corrected = cv2.addWeighted(overlay, 0.16, corrected, 0.84, 0.0)
            for segment in evidence.boundary_segments:
                if segment.status == "VISIBLE":
                    cv2.line(
                        corrected, tuple(map(int, segment.image_start)),
                        tuple(map(int, segment.image_end)), color, 9, cv2.LINE_AA,
                    )
                else:
                    _draw_dashed_line(
                        corrected, segment.image_start, segment.image_end, color, 4
                    )
            cv2.putText(
                corrected,
                f"ZICHTBAAR {patch.patch_id} | {evidence.frame_coverage:.0%} frame | "
                f"{len(evidence.boundary_segments)} grenssegmenten",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4, cv2.LINE_AA,
            )
            panels.append(cv2.resize(corrected, (960, 540)))
        if atlas.manual_midfield_line is not None:
            midfield = atlas.manual_midfield_line
            capture.set(cv2.CAP_PROP_POS_FRAMES, midfield.frame_number)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Middenlijnframe {midfield.frame_number} kon niet worden gelezen.")
            corrected = cv2.undistort(frame, lens.camera_matrix, lens.distortion_coefficients)
            corrected_points = lens.undistort_points(
                np.asarray(midfield.points, dtype=np.float64)
            )
            line = _fit_line(corrected_points)
            first, second = _line_frame_endpoints(
                line, corrected.shape[1], corrected.shape[0]
            )
            cv2.line(corrected, first, second, (255, 255, 0), 9, cv2.LINE_AA)
            for point in corrected_points:
                cv2.circle(
                    corrected, tuple(np.rint(point).astype(int)), 9,
                    (255, 0, 255), -1, cv2.LINE_AA,
                )
            cv2.putText(
                corrected,
                "11v11-MIDDENLIJN | EXPLICIETE RICHTING VOOR 8v8-ZIJLIJNEN",
                (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.35,
                (255, 255, 0), 4, cv2.LINE_AA,
            )
            panels.append(cv2.resize(corrected, (960, 540)))
    finally:
        capture.release()
    if not panels or not cv2.imwrite(str(path), np.hstack(panels)):
        raise RuntimeError("Atlaspreview kon niet worden opgeslagen.")


def _goal_vanishing_points(video, structures, perspective, lens, seeds, parallel_reference):
    by_label = {item.label: item for item in perspective.views}
    by_goal = {item.goal_id: item for item in structures}
    capture = cv2.VideoCapture(str(video))
    result = {}
    diagnostics = {}
    try:
        for goal_id, label in (("A", "left_goal"), ("B", "right_goal")):
            view = by_label[label]
            structure = by_goal[goal_id]
            transform = np.eye(3, dtype=np.float64)
            if view.frame_number != structure.frame_number:
                source = _read_frame(capture, view.frame_number)
                target = _read_frame(capture, structure.frame_number)
                try:
                    edge = estimate_ground_frame_edge(label, goal_id, source, target)
                except ValueError:
                    edge = estimate_frame_edge(label, goal_id, source, target)
                transform = edge.source_to_target
            equations_by_direction = {
                PerspectiveDirection.BETWEEN_GOALS: [],
                PerspectiveDirection.ALONG_END_LINES: [],
            }
            for line in view.lines:
                if line.direction is PerspectiveDirection.UNKNOWN:
                    continue
                points = np.asarray(line.points, dtype=np.float64)
                mapped = cv2.perspectiveTransform(points.reshape(1, -1, 2), transform).reshape(-1, 2)
                corrected = lens.undistort_points(mapped)
                equations_by_direction[line.direction].append(_fit_line(corrected))
            candidates = []
            observed = lens.undistort_points(
                np.asarray(seeds[goal_id].front_sideline_observations, dtype=np.float64)
            )
            observed_line = _fit_line(observed)
            for direction, equations in equations_by_direction.items():
                if len(equations) < 2:
                    continue
                _u, _s, vh = np.linalg.svd(np.asarray(equations, dtype=np.float64))
                point = vh[-1]
                if abs(float(point[2])) < 1e-9:
                    continue
                point /= point[2]
                candidates.append((abs(float(observed_line @ point)), direction, point))
            if not candidates:
                raise RuntimeError(f"Doel {goal_id} mist een bruikbaar 11v11-verdwijnpunt.")
            _distance, _direction, point = min(candidates, key=lambda item: item[0])
            result[goal_id] = tuple(point[:2].tolist())
            explicit_lines = tuple(
                line for line in parallel_reference.lines
                if line.line_type in ("goal_area_5m", "penalty_area_16m")
                and line.frame_number == structure.frame_number
            )
            if len(explicit_lines) == 2:
                equations = []
                for line in explicit_lines:
                    corrected = lens.undistort_points(
                        np.asarray(line.points, dtype=np.float64)
                    )
                    equations.append(_fit_line(corrected))
                explicit_point = np.cross(equations[0], equations[1])
                if abs(float(explicit_point[2])) > 1e-9:
                    explicit_point /= explicit_point[2]
                    difference = float(np.linalg.norm(explicit_point[:2] - point[:2]))
                    focal = max(float(lens.focal_length_px), 1.0)
                    diagnostics[goal_id] = {
                        "difference_px": difference,
                        "difference_degrees": float(np.degrees(np.arctan2(difference, focal))),
                        "automatic_vanishing_point": tuple(point[:2].tolist()),
                        "explicit_vanishing_point": tuple(explicit_point[:2].tolist()),
                    }
                    result[goal_id] = tuple(explicit_point[:2].tolist())
    finally:
        capture.release()
    return result, diagnostics


def _read_frame(capture, frame_number):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Videoframe {frame_number} kon niet worden gelezen.")
    return frame


def _fit_line(points):
    center = np.mean(points, axis=0)
    _u, _s, vh = np.linalg.svd(points - center)
    direction = vh[0]
    normal = np.asarray((-direction[1], direction[0]))
    equation = np.asarray((normal[0], normal[1], -float(normal @ center)))
    return equation / np.linalg.norm(equation[:2])


def _line_frame_endpoints(line, width, height):
    a, b, c = map(float, line)
    points = []
    if abs(b) > 1e-9:
        for x in (0.0, float(width - 1)):
            y = -(a * x + c) / b
            if 0.0 <= y < height:
                points.append((x, y))
    if abs(a) > 1e-9:
        for y in (0.0, float(height - 1)):
            x = -(b * y + c) / a
            if 0.0 <= x < width:
                points.append((x, y))
    unique = []
    for point in points:
        if not any(np.linalg.norm(np.asarray(point) - existing) < 1.0 for existing in unique):
            unique.append(np.asarray(point))
    if len(unique) < 2:
        raise RuntimeError("De 11v11-middenlijn kan niet over het beeld worden getekend.")
    return tuple(np.rint(unique[0]).astype(int)), tuple(np.rint(unique[1]).astype(int))


def _bind_vanishing_point_to_observed_sideline(vanishing, corner, observed_points):
    """Keep the global perspective prior on the actually observed touchline."""
    points = np.vstack((np.asarray(corner, dtype=np.float64), observed_points))
    center = np.mean(points, axis=0)
    _u, _s, vh = np.linalg.svd(points - center)
    direction = vh[0]
    prior = np.asarray(vanishing, dtype=np.float64)
    projected = center + direction * float((prior - center) @ direction)
    return tuple(map(float, projected))


def _orient_ground_homography(homography, goal_id, seed, profile, lens):
    observations = seed.front_sideline_observations
    if seed.front_corner is None or not observations:
        return homography, False
    corner = lens.undistort_points(np.asarray((seed.front_corner,), dtype=np.float64))[0]
    observed_points = lens.undistort_points(np.asarray(observations, dtype=np.float64))
    distances = np.linalg.norm(observed_points - corner, axis=1)
    observed = observed_points[int(np.argmax(distances))] - corner
    interior_x = 10.0 if goal_id == "A" else profile.pitch_length_m - 10.0
    corner_x = 0.0 if goal_id == "A" else profile.pitch_length_m
    sample = np.asarray(((corner_x, profile.pitch_width_m), (interior_x, profile.pitch_width_m)))

    def direction(matrix):
        homogeneous = np.column_stack((sample, np.ones(2)))
        projected = (matrix @ homogeneous.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        return projected[1] - projected[0]

    reflection = np.asarray(
        ((-1.0, 0.0, 2.0 * corner_x), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    reflected = homography @ reflection
    original_score = float(direction(homography) @ observed)
    reflected_score = float(direction(reflected) @ observed)
    return (reflected, True) if reflected_score > original_score else (homography, False)


def _draw_dashed_line(image, start, end, color, thickness):
    first = np.asarray(start, dtype=np.float64)
    second = np.asarray(end, dtype=np.float64)
    length = float(np.linalg.norm(second - first))
    if length < 1.0:
        return
    direction = (second - first) / length
    position = 0.0
    while position < length:
        dash_end = min(position + 22.0, length)
        a = first + direction * position
        b = first + direction * dash_end
        cv2.line(
            image, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
            color, thickness, cv2.LINE_AA,
        )
        position += 36.0


if __name__ == "__main__":
    main()
