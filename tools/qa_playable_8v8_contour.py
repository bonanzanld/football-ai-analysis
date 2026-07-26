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
from football_ai.calibration.bootstrap.goal_seed import load_goal_seeds
from football_ai.calibration.camera_anchor_bank_3d import load_camera_anchor_bank
from football_ai.calibration.geometry_validation import validate_projected_pitch_geometry
from football_ai.calibration.playable_boundary_semantics import (
    BoundaryEvidenceSource,
    PlayableBoundaryBinding,
    PlayableBoundaryRole,
)
from football_ai.calibration.playable_field_contour import (
    create_playable_field_contour,
    validate_playable_contour_geometry,
)
from football_ai.calibration.perspective_parallelism import (
    assess_playable_sideline_parallelism,
    correct_sidelines_to_ground_perpendicular,
    detect_long_white_right_reference,
    estimate_vanishing_point_from_lines,
    measure_ground_line_angle,
    rebuild_from_confirmed_backline_and_ground_horizon,
    rebuild_from_endline_goal_area_and_far_support,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="QA voor de definitieve metrische 8v8-speelveldcontour.")
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4")
    parser.add_argument("--format", choices=("8v8",), default="8v8")
    parser.add_argument(
        "--confirm-contour",
        action="store_true",
        help="Bevestig na visuele QA dat de vier gele speelveldgrenzen correct over het veld lopen.",
    )
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    binding_report = json.loads((output_dir / f"{prefix}_shared_end_lines_qa.json").read_text())
    bindings = []
    for assessment in binding_report["assessments"]:
        item = assessment["binding"]
        bindings.append(
            PlayableBoundaryBinding(
                PlayableBoundaryRole(item["role"]),
                BoundaryEvidenceSource(item["source"]),
                item["source_id"],
                bool(item["confirmed"]),
            )
        )
    profile = create_detection_profile(args.format)
    contour = create_playable_field_contour(profile, tuple(bindings))
    seeds = {
        item.goal_id: item
        for item in load_goal_seeds(output_dir / f"{prefix}_goal_seeds.json")
    }
    quality = validate_playable_contour_geometry(
        contour.polygon_ground_m,
        contour.pitch_length_m,
        contour.pitch_width_m,
    )
    bank = load_camera_anchor_bank(output_dir / f"{prefix}_camera_anchors_3d.json")
    capture = cv2.VideoCapture(str(video))
    previews = []
    projection_quality = {}
    parallelism_quality = {}
    support_alignment_quality = {}
    orthogonality_quality = {}
    for anchor in sorted(bank.anchors, key=lambda item: item.anchor_id):
        capture.set(cv2.CAP_PROP_POS_FRAMES, anchor.frame_number)
        success, frame = capture.read()
        if not success:
            continue
        original_polygon = contour.project(anchor.projection.ground_homography())
        line_report_name = (
            f"{prefix}_global_frame_graph_ground_qa.json"
            if anchor.anchor_id == "goal-a"
            else f"{prefix}_global_frame_graph_ground_goal-b_qa.json"
        )
        line_report = json.loads((output_dir / line_report_name).read_text(encoding="utf-8"))
        transverse_clusters = line_report["line_diversity"]["transverse"]["clusters"]
        reference_lines = tuple(
            (
                tuple(cluster["representative"]["start"]),
                tuple(cluster["representative"]["end"]),
            )
            for cluster in transverse_clusters
        )
        polygon = original_polygon.copy()
        parallelism = None
        refined_homography = None
        support_vanishing_point = None
        seed = seeds[anchor.goal_id]
        goal_area_line = (
            (seed.front_sideline_support, seed.front_sideline_support_end)
            if seed.front_sideline_support is not None
            and seed.front_sideline_support_end is not None
            else _select_goal_area_line(line_report, transverse_clusters)
        )
        if (
            seed.front_sideline_support is None
            or seed.front_sideline_support_end is None
            or goal_area_line is None
        ):
            support_alignment_quality[anchor.anchor_id] = {
                "confirmed": False,
                "reason": "Twee punten op de witte 5,5-meterlijn/hoedjeslijn zijn vereist.",
            }
        else:
            rear_index, opposite_rear = (
                (0, 1) if anchor.goal_id == "A" else (1, 0)
            )
            automatic_far_support = tuple(map(float, original_polygon[opposite_rear]))
            try:
                if anchor.goal_id == "B":
                    confirmed_backline = (
                        tuple(map(float, original_polygon[1])),
                        tuple(map(float, original_polygon[2])),
                    )
                    observations = (
                        seed.front_sideline_observations
                        if len(seed.front_sideline_observations) >= 2
                        else (seed.front_sideline_support, seed.front_sideline_support_end)
                    )
                    goal_area_line, anchored_rms = _fit_line_through_anchor(
                        confirmed_backline[1],
                        observations,
                    )
                    white_parallel_reference = detect_long_white_right_reference(frame)
                    if white_parallel_reference is None:
                        raise ValueError(
                            "Lange witte 11v11-referentielijn rechts niet betrouwbaar gevonden."
                        )
                    polygon, support_vanishing_point = (
                        rebuild_from_confirmed_backline_and_ground_horizon(
                            polygon,
                            anchor.goal_id,
                            confirmed_backline,
                            goal_area_line,
                            anchor.projection.ground_homography(),
                            parallel_reference_line=white_parallel_reference,
                        )
                    )
                else:
                    polygon, support_vanishing_point = rebuild_from_endline_goal_area_and_far_support(
                        polygon,
                        anchor.goal_id,
                        (seed.first_ground, seed.second_ground),
                        goal_area_line,
                        automatic_far_support,
                    )
            except ValueError as error:
                support_alignment_quality[anchor.anchor_id] = {
                    "confirmed": False,
                    "reason": str(error),
                }
            else:
                if anchor.goal_id == "B":
                    front_support_distance = anchored_rms
                    support_threshold = 30.0
                else:
                    front_support_distance = _point_line_distance(
                        seed.front_sideline_support,
                        goal_area_line,
                    )
                    support_threshold = 20.0
                support_alignment_quality[anchor.anchor_id] = {
                    "confirmed": front_support_distance <= support_threshold,
                    "rear_support": list(automatic_far_support),
                    "rear_support_origin": (
                        "confirmed_42_5m_backline_and_ground_horizon"
                        if anchor.goal_id == "B"
                        else "derived_from_ground_plane"
                    ),
                    "front_support": list(seed.front_sideline_support),
                    "front_support_distance_px": front_support_distance,
                    "derived_vanishing_point": list(support_vanishing_point),
                    "reason": (
                        ""
                        if front_support_distance <= support_threshold
                        else "De globale hoedjes-/lijnpunten wijken te sterk af van de verankerde gemiddelde zijlijn."
                    ),
                    "near_sideline_binding": {
                        "role": "near_sideline",
                        "source": "full_pitch_goal_area_line",
                        "source_id": "11v11_goal_area_5_5m",
                        "confirmed": True,
                        "confirmation_origin": "operator_configuration",
                    },
                }
                if anchor.goal_id == "B":
                    support_alignment_quality[anchor.anchor_id]["white_parallel_reference"] = [
                        list(white_parallel_reference[0]),
                        list(white_parallel_reference[1]),
                    ]
        backline_pair = (
            (tuple(polygon[0]), tuple(polygon[3]))
            if anchor.goal_id == "A"
            else (tuple(polygon[1]), tuple(polygon[2]))
        )
        front_sideline_pair = (tuple(polygon[3]), tuple(polygon[2]))
        ground_angle_before = measure_ground_line_angle(
            backline_pair,
            front_sideline_pair,
            anchor.projection.ground_homography(),
        )
        corrected_to_metric = False
        orthogonality_tolerance = 8.0
        if abs(90.0 - ground_angle_before) > orthogonality_tolerance:
            polygon = correct_sidelines_to_ground_perpendicular(
                polygon,
                anchor.goal_id,
                anchor.projection.ground_homography(),
            )
            corrected_to_metric = True
            support_vanishing_point = estimate_vanishing_point_from_lines(
                (
                    (tuple(polygon[0]), tuple(polygon[1])),
                    (tuple(polygon[3]), tuple(polygon[2])),
                )
            )
            observations = (
                seed.front_sideline_observations
                if seed.front_sideline_observations
                else (seed.front_sideline_support, seed.front_sideline_support_end)
            )
            residuals = [
                _point_line_distance(point, (tuple(polygon[3]), tuple(polygon[2])))
                for point in observations
                if point is not None
            ]
            observation_rms = float(np.sqrt(np.mean(np.square(residuals))))
            support_alignment_quality[anchor.anchor_id].update(
                {
                    "confirmed": observation_rms <= 30.0,
                    "front_support_distance_px": observation_rms,
                    "reason": (
                        ""
                        if observation_rms <= 30.0
                        else "Hoedjes-/lijnpunten wijken te sterk af van de metrisch haakse zijlijn."
                    ),
                }
            )
        corrected_backline_pair = (
            (tuple(polygon[0]), tuple(polygon[3]))
            if anchor.goal_id == "A"
            else (tuple(polygon[1]), tuple(polygon[2]))
        )
        ground_angle_after = measure_ground_line_angle(
            corrected_backline_pair,
            (tuple(polygon[3]), tuple(polygon[2])),
            anchor.projection.ground_homography(),
        )
        orthogonality_quality[anchor.anchor_id] = {
            "confirmed": abs(90.0 - ground_angle_after) <= orthogonality_tolerance,
            "angle_before_degrees": ground_angle_before,
            "angle_after_degrees": ground_angle_after,
            "maximum_deviation_degrees": orthogonality_tolerance,
            "corrected_to_metric_projection": corrected_to_metric,
        }
        if support_vanishing_point is not None:
            parallelism = assess_playable_sideline_parallelism(
                polygon,
                np.asarray(support_vanishing_point, dtype=np.float64),
            )
            refined_homography = cv2.getPerspectiveTransform(
                contour.polygon_ground_m.astype(np.float32),
                polygon.astype(np.float32),
            )
            parallelism_quality[anchor.anchor_id] = {
                "reference_line_count": len(reference_lines),
                "confirmed": parallelism.valid,
                "after": parallelism.to_dict(),
                "refined_ground_homography": refined_homography.tolist(),
                "direction_source": "operator_confirmed_5_5m_line",
                "reason": "" if parallelism.valid else "De twee speelveldzijlijnen delen niet hetzelfde verdwijnpunt.",
            }
            if len(reference_lines) >= 2:
                automatic_vanishing_point = estimate_vanishing_point_from_lines(reference_lines)
                automatic_check = assess_playable_sideline_parallelism(
                    polygon,
                    automatic_vanishing_point,
                )
                parallelism_quality[anchor.anchor_id]["automatic_white_line_check"] = {
                    "informational_only": True,
                    "result": automatic_check.to_dict(),
                    "warning": (
                        ""
                        if automatic_check.valid
                        else "Automatische witte-lijndetectie wijkt af van de bevestigde 5,5m-lijn."
                    ),
                }
            else:
                parallelism_quality[anchor.anchor_id]["automatic_white_line_check"] = {
                    "informational_only": True,
                    "warning": "Te weinig automatisch herkende witte lijnen voor een onafhankelijke controle.",
                }
        else:
            parallelism_quality[anchor.anchor_id] = {
                "reference_line_count": len(reference_lines),
                "confirmed": False,
                "reason": "Geen geldige handmatig bevestigde 5,5m-/hoedjeslijn beschikbaar.",
            }
        image_quality = validate_projected_pitch_geometry(
            polygon,
            frame.shape[1],
            frame.shape[0],
        )
        frame_polygon = np.asarray(
            ((0.0, 0.0), (frame.shape[1] - 1.0, 0.0), (frame.shape[1] - 1.0, frame.shape[0] - 1.0), (0.0, frame.shape[0] - 1.0)),
            dtype=np.float32,
        )
        intersection_area, _intersection = cv2.intersectConvexConvex(
            polygon.astype(np.float32),
            frame_polygon,
        )
        visible_frame_coverage = float(intersection_area / max(frame.shape[0] * frame.shape[1], 1))
        visible_polygon_fraction = float(
            intersection_area / max(image_quality.polygon_area_pixels, 1.0)
        )
        projection_quality[anchor.anchor_id] = {
            "valid": image_quality.valid,
            "polygon_area_pixels": image_quality.polygon_area_pixels,
            "frame_area_ratio": image_quality.frame_area_ratio,
            "minimum_edge_length_pixels": image_quality.minimum_edge_length_pixels,
            "visible_frame_coverage": visible_frame_coverage,
            "visible_polygon_fraction": visible_polygon_fraction,
            "errors": list(image_quality.errors),
        }
        preview = frame.copy()
        finite = np.all(np.isfinite(polygon), axis=1)
        if np.all(finite):
            cv2.polylines(
                preview,
                [np.round(original_polygon).astype(np.int32)],
                True,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            overlay = preview.copy()
            cv2.fillPoly(overlay, [np.round(polygon).astype(np.int32)], (30, 170, 30))
            preview = cv2.addWeighted(overlay, 0.20, preview, 0.80, 0.0)
            cv2.polylines(preview, [np.round(polygon).astype(np.int32)], True, (0, 255, 255), 5, cv2.LINE_AA)
            for corner, point in zip(contour.corners, polygon):
                position = tuple(np.round(point).astype(int))
                cv2.circle(preview, position, 8, (255, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(preview, corner.corner_id, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2, cv2.LINE_AA)
            if anchor.goal_id == "B" and "white_parallel_reference" in support_alignment_quality[anchor.anchor_id]:
                reference = support_alignment_quality[anchor.anchor_id]["white_parallel_reference"]
                cv2.line(
                    preview,
                    tuple(np.round(reference[0]).astype(int)),
                    tuple(np.round(reference[1]).astype(int)),
                    (255, 255, 0),
                    3,
                    cv2.LINE_AA,
                )
        geometry_valid = quality.valid and image_quality.valid
        direction_confirmed = bool(parallelism_quality[anchor.anchor_id]["confirmed"])
        status = (
            "BEVESTIGD"
            if geometry_valid and direction_confirmed and args.confirm_contour
            else "RICHTING BEVESTIGD - VISUEEL CONTROLEREN"
            if geometry_valid and direction_confirmed
            else "ONVOLDOENDE LIJNREFERENTIES"
            if geometry_valid
            else "ONGELDIG"
        )
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 66), (20, 20, 20), -1)
        cv2.putText(
            preview,
            f"8v8 SPEELVELD 64x42.5m | {anchor.anchor_id} | {status}",
            (14, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 220, 40) if geometry_valid and direction_confirmed and args.confirm_contour else (0, 220, 255) if geometry_valid else (0, 80, 255),
            2,
            cv2.LINE_AA,
        )
        previews.append(preview)
    capture.release()
    preview_path = output_dir / f"{prefix}_playable_field_contour_qa.jpg"
    cv2.imwrite(str(preview_path), np.hstack(previews))
    report_path = output_dir / f"{prefix}_playable_field_contour_qa.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contour": contour.to_dict(),
                "geometry_quality": quality.to_dict(),
                "projection_quality": projection_quality,
                "parallelism_quality": parallelism_quality,
                "support_alignment_quality": support_alignment_quality,
                "orthogonality_quality": orthogonality_quality,
                "operator_confirmed": args.confirm_contour,
                "status": (
                    "PASS"
                    if args.confirm_contour and quality.valid and all(item["valid"] for item in projection_quality.values()) and all(item["confirmed"] for item in parallelism_quality.values()) and all(item["confirmed"] for item in support_alignment_quality.values()) and all(item["confirmed"] for item in orthogonality_quality.values())
                    else "WARNING"
                    if quality.valid and all(item["valid"] for item in projection_quality.values()) and all(item["confirmed"] for item in parallelism_quality.values()) and all(item["confirmed"] for item in support_alignment_quality.values()) and all(item["confirmed"] for item in orthogonality_quality.values())
                    else "FAIL"
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    overall_valid = quality.valid and all(item["valid"] for item in projection_quality.values())
    all_directions_confirmed = all(item["confirmed"] for item in parallelism_quality.values())
    all_supports_confirmed = all(item["confirmed"] for item in support_alignment_quality.values())
    all_angles_confirmed = all(item["confirmed"] for item in orthogonality_quality.values())
    status = (
        "PASS"
        if overall_valid and all_directions_confirmed and all_supports_confirmed and all_angles_confirmed and args.confirm_contour
        else "WARNING"
        if overall_valid and all_directions_confirmed and all_supports_confirmed and all_angles_confirmed
        else "FAIL"
    )
    print(f"Contour: {status} | oppervlakte {quality.area_m2:.1f}m2 | max afmetingsfout {quality.maximum_dimension_error_m:.2f}m")
    for anchor_id, item in projection_quality.items():
        print(
            f"  {anchor_id}: {'GELDIG' if item['valid'] else 'ONGELDIG'} | "
            f"zichtbaar velddeel {item['visible_frame_coverage']:.1%} van frame | "
            f"{item['visible_polygon_fraction']:.1%} van contour zichtbaar"
        )
        direction = parallelism_quality[anchor_id]
        support = support_alignment_quality[anchor_id]
        print(
            f"    hoek/hoedjeslijn: {'BEVESTIGD' if support['confirmed'] else 'ONVOLDOENDE'} | "
            f"{support.get('reason', '')}"
        )
        if direction["confirmed"]:
            print(
                f"    paralleliteit: BEVESTIGD | max fout "
                f"{direction['after']['maximum_residual_degrees']:.2f} graden"
            )
        else:
            print(f"    paralleliteit: ONVOLDOENDE | {direction.get('reason', '')}")
        angle = orthogonality_quality[anchor_id]
        print(
            f"    grondhoek: {angle['angle_after_degrees']:.2f} graden | "
            f"marge +/-{angle['maximum_deviation_degrees']:.1f} graden"
        )
        print(
            f"    lijnsteun RMS: {support.get('front_support_distance_px', float('nan')):.1f} px"
        )
    print(f"QA-preview: {preview_path}")
    print(f"QA-rapport: {report_path}")


def _select_goal_area_line(
    line_report: dict,
    transverse_clusters: list[dict],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not transverse_clusters:
        return None
    match = line_report.get("full_pitch_goal_zone_match")
    selected = None
    if match and match.get("resolved") and "goal_area" in match.get("marking_ids", []):
        index = match["marking_ids"].index("goal_area")
        target_offset = float(match["detected_offsets_m"][index])
        selected = min(
            transverse_clusters,
            key=lambda item: abs(float(item["mean_ground_offset_m"]) - target_offset),
        )
    elif len(transverse_clusters) == 1:
        selected = transverse_clusters[0]
    if selected is None:
        return None
    representative = selected["representative"]
    return tuple(representative["start"]), tuple(representative["end"])


def _point_line_distance(
    point: tuple[float, float],
    line: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    start, end = np.asarray(line[0]), np.asarray(line[1])
    direction = end - start
    offset = np.asarray(point) - start
    cross = direction[0] * offset[1] - direction[1] * offset[0]
    return abs(float(cross)) / max(float(np.linalg.norm(direction)), 1e-9)


def _fit_line_through_anchor(
    anchor: tuple[float, float],
    observations: tuple[tuple[float, float], ...],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], float]:
    """Fit a tolerant direction while keeping the confirmed field corner fixed."""
    origin = np.asarray(anchor, dtype=np.float64)
    samples = np.asarray(observations, dtype=np.float64)
    offsets = samples - origin
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] != 2:
        raise ValueError("Minimaal twee globale zijlijnaanwijzingen vereist.")
    _u, singular_values, axes = np.linalg.svd(offsets, full_matrices=False)
    if singular_values[0] < 40.0:
        raise ValueError("De zijlijnaanwijzingen liggen te dicht bij de bevestigde hoek.")
    direction = axes[0]
    projections = offsets @ direction
    if float(np.mean(projections)) < 0.0:
        direction = -direction
        projections = -projections
    residuals = offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0]
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    extent = max(float(np.max(projections)), 100.0)
    end = origin + direction * extent
    return (tuple(map(float, origin)), tuple(map(float, end))), rms


if __name__ == "__main__":
    main()
