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
from football_ai.calibration.global_frame_graph import (
    AbsoluteGroundConstraint,
    AbsoluteGroundPointConstraint,
    FrameGraphNode,
    GroundDirectionConstraint,
    estimate_ground_frame_edge,
    select_maximum_quality_tree,
    select_cycle_consistent_edges,
    solve_global_frame_graph,
)
from football_ai.calibration.full_pitch_markings import (
    circle_center_matches_halfway_line,
    create_standard_full_pitch_marking_model,
    match_marking_offsets,
)
from football_ai.calibration.ground_line_evidence import GroundLineFamily, detect_metric_ground_lines
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
from football_ai.calibration.global_ground_registration import (
    GlobalGroundRegistration,
    RegisteredGroundFrame,
    save_global_ground_registration,
)
from football_ai.calibration.goal_zone_markings import (
    create_goal_zone_reference,
    match_goal_zone_depth_lines,
)
from football_ai.calibration.ground_circle_evidence import (
    GroundCircleEvidence,
    detect_metric_center_circle,
    estimate_circle_consensus,
    project_ground_circle,
    validate_ground_circle_on_frame,
)
from football_ai.calibration.orthogonal_ground_orientation import (
    ImageLineObservation,
    cluster_physical_lines,
    estimate_orthogonal_ground_orientation,
    summarize_line_diversity,
    transform_line_observation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Globale framegraph-QA voor het loodrechte 2D-grondvlak.")
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--start", type=float, default=900.0)
    parser.add_argument("--duration", type=float, default=110.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--minimum-length", type=float, default=3.0)
    parser.add_argument(
        "--reference-anchor",
        choices=("goal-a", "goal-b"),
        default="goal-a",
        help="Primair 3D-camera-anker waarnaar de framegraph wordt uitgelijnd.",
    )
    args = parser.parse_args()

    video = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    bank = load_camera_anchor_bank(output_dir / f"{prefix}_camera_anchors_3d.json")
    reference_anchor = next(item for item in bank.anchors if item.anchor_id == args.reference_anchor)
    profile = create_detection_profile(args.format)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video kon niet worden geopend: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    times = list(np.arange(args.start, args.start + args.duration + 1e-6, args.interval))
    times.append(reference_anchor.time_seconds)
    times = sorted(set(round(float(item), 6) for item in times))
    nodes, frames = [], {}
    for time_seconds in times:
        frame_number = int(round(time_seconds * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
        if not success:
            continue
        node_id = f"f{frame_number}"
        nodes.append(FrameGraphNode(node_id, frame_number, time_seconds))
        frames[node_id] = frame
    capture.release()
    reference_id = f"f{reference_anchor.frame_number}"
    ordered = tuple(nodes)
    edges = []
    attempted = 0
    for source_index, source in enumerate(ordered):
        for gap in (1, 2):
            target_index = source_index + gap
            if target_index >= len(ordered):
                continue
            target = ordered[target_index]
            attempted += 1
            try:
                edges.append(
                    estimate_ground_frame_edge(
                        source.node_id,
                        target.node_id,
                        frames[source.node_id],
                        frames[target.node_id],
                    )
                )
            except ValueError:
                pass
        if source_index and source_index % 20 == 0:
            print(f"Graphverbindingen: {source_index}/{len(ordered)} nodes verwerkt")
    tree_edges = select_maximum_quality_tree(tuple(ordered), tuple(edges))
    tree_solution = solve_global_frame_graph(
        tuple(ordered),
        tree_edges,
        reference_id,
        _pruning_rounds=0,
    )
    consistent_edges = select_cycle_consistent_edges(
        tree_solution,
        tree_edges,
        tuple(edges),
    )
    direction_constraints = []
    manual_direction_constraints = []
    absolute_ground_constraints = []
    absolute_ground_point_constraints = []
    reference_ground_to_image = reference_anchor.projection.ground_homography()
    manual_line_axes: dict[str, tuple[int, ...]] = {}
    manual_path = output_dir / f"{prefix}_manual_perspective_reference.json"
    manual_reference = load_manual_perspective_reference(manual_path) if manual_path.exists() else None
    manual_assignment = None
    manual_metric_reference_applied = False
    if manual_reference is not None:
        left_view = next((view for view in manual_reference.views if view.label == "left_goal"), None)
        left_anchor = next((item for item in bank.anchors if item.anchor_id == "goal-a"), None)
        if left_view is not None and left_anchor is not None and len(left_view.lines) >= 2:
            left_path = output_dir / f"{prefix}_view_{left_anchor.goal_id.upper()}_3d.json"
            left_data = json.loads(left_path.read_text(encoding="utf-8"))
            left_observations = CameraViewObservations.from_dict(left_data["view"])
            left_reference = create_field_reference_3d(profile)
            left_ground_observations = left_observations.ground_observations(left_reference)
            left_ground_points = np.asarray(
                [left_reference.landmark(item.landmark_id).point.as_tuple()[:2] for item in left_ground_observations],
                dtype=np.float64,
            )
            left_image_points = np.asarray(
                [item.image_point for item in left_ground_observations], dtype=np.float64
            )
            left_refinement = refine_ground_homography_with_lines(
                left_anchor.projection.ground_homography(),
                left_ground_points,
                left_image_points,
                tuple(item.equation() for item in left_view.lines),
            )
            left_node_id = f"f{left_view.frame_number}"
            manual_line_axes[left_view.label] = left_refinement.line_axis_assignment
            absolute_ground_point_constraints.append(
                AbsoluteGroundPointConstraint(
                    left_node_id,
                    left_ground_points,
                    left_image_points,
                    5000.0,
                )
            )
            print(
                "Doel A-richtingen bepaald met handmatige lijnen: "
                f"punt-RMS {left_refinement.rms_point_error_px:.2f}px, "
                f"lijn-RMS {left_refinement.rms_line_error_px:.2f}px."
            )
        complete_views = tuple(view for view in manual_reference.views if view.perspective_complete)
        if complete_views:
            manual_view = complete_views[0]
            manual_anchor_id = {"left_goal": "goal-a", "right_goal": "goal-b"}.get(manual_view.label)
            if manual_anchor_id is not None:
                manual_anchor = next(item for item in bank.anchors if item.anchor_id == manual_anchor_id)
                goal_id = manual_anchor.goal_id.upper()
                observation_path = output_dir / f"{prefix}_view_{goal_id}_3d.json"
                observation_data = json.loads(observation_path.read_text(encoding="utf-8"))
                view_observations = CameraViewObservations.from_dict(observation_data["view"])
                field_reference = create_field_reference_3d(profile)
                ground_observations = view_observations.ground_observations(field_reference)
                ground_points = np.asarray(
                    [field_reference.landmark(item.landmark_id).point.as_tuple()[:2] for item in ground_observations],
                    dtype=np.float64,
                )
                image_points = np.asarray([item.image_point for item in ground_observations], dtype=np.float64)
                manual_vanishing = (
                    manual_view.vanishing_point(PerspectiveDirection.BETWEEN_GOALS),
                    manual_view.vanishing_point(PerspectiveDirection.ALONG_END_LINES),
                )
                refinement = refine_ground_homography_with_vanishing_points(
                    manual_anchor.projection.ground_homography(),
                    ground_points,
                    image_points,
                    manual_vanishing,
                )
                manual_node_id = f"f{manual_view.frame_number}"
                if manual_node_id in tree_solution.node_to_reference:
                    absolute_ground_constraints.append(
                        AbsoluteGroundConstraint(manual_node_id, refinement.homography, 8.0)
                    )
                    manual_to_reference = tree_solution.node_to_reference[manual_node_id]
                    reference_ground_to_image = manual_to_reference @ refinement.homography
                    manual_assignment = refinement.direction_assignment
                    manual_metric_reference_applied = len(ground_observations) >= 3
                    print(
                        f"Handmatige perspectiefreferentie toegepast via {manual_view.label}: "
                        f"grondpunten RMS {refinement.rms_point_error_px:.2f}px, "
                        f"max {refinement.maximum_point_error_px:.2f}px."
                    )
    for node in ordered:
        node_to_reference = tree_solution.node_to_reference.get(node.node_id)
        if node_to_reference is None:
            continue
        ground_to_node = np.linalg.inv(node_to_reference) @ reference_ground_to_image
        detection = detect_metric_ground_lines(
            frames[node.node_id],
            profile,
            ground_to_node,
            args.minimum_length,
        )
        for line in detection.lines:
            axis = 0 if line.family is GroundLineFamily.LONGITUDINAL else 1
            vanishing = ground_to_node[:, axis]
            if abs(float(vanishing[2])) < 1e-9:
                continue
            vanishing_point = vanishing[:2] / vanishing[2]
            start = np.asarray(line.image_start, dtype=np.float64)
            end = np.asarray(line.image_end, dtype=np.float64)
            direction = end - start
            toward_vanishing = vanishing_point - 0.5 * (start + end)
            denominator = float(np.linalg.norm(direction) * np.linalg.norm(toward_vanishing))
            if denominator < 1e-9:
                continue
            angular_error = float(
                np.degrees(
                    np.arccos(
                        np.clip(abs(float(direction @ toward_vanishing)) / denominator, 0.0, 1.0)
                    )
                )
            )
            if angular_error > 3.5:
                continue
            direction_constraints.append(
                GroundDirectionConstraint(
                    node.node_id,
                    line.family,
                    line.image_start,
                    line.image_end,
                    float(np.clip(line.confidence * np.sqrt(line.metric_length / 3.0), 0.2, 2.0)),
                )
            )
    if manual_reference is not None and manual_assignment is not None:
        target_to_axis = {target: axis for axis, target in enumerate(manual_assignment)}
        for view in manual_reference.views:
            node_id = f"f{view.frame_number}"
            node_to_reference = tree_solution.node_to_reference.get(node_id)
            if node_to_reference is None:
                continue
            ground_to_node = np.linalg.inv(node_to_reference) @ reference_ground_to_image
            for line_index, line in enumerate(view.lines):
                if line.direction is PerspectiveDirection.UNKNOWN:
                    predefined = manual_line_axes.get(view.label)
                    if predefined is not None and line_index < len(predefined):
                        axis = predefined[line_index]
                    else:
                        errors = []
                        points = np.asarray(line.points, dtype=np.float64)
                        direction = points[-1] - points[0]
                        midpoint = np.mean(points, axis=0)
                        for axis_candidate in (0, 1):
                            vanishing = ground_to_node[:, axis_candidate]
                            if abs(float(vanishing[2])) < 1e-9:
                                errors.append(90.0)
                                continue
                            toward = vanishing[:2] / vanishing[2] - midpoint
                            denominator = float(np.linalg.norm(direction) * np.linalg.norm(toward))
                            cosine = 0.0 if denominator < 1e-9 else np.clip(abs(float(direction @ toward)) / denominator, 0.0, 1.0)
                            errors.append(float(np.degrees(np.arccos(cosine))))
                        axis = int(np.argmin(errors))
                        if errors[axis] > 12.0:
                            continue
                else:
                    target = 0 if line.direction is PerspectiveDirection.BETWEEN_GOALS else 1
                    axis = target_to_axis[target]
                family = GroundLineFamily.LONGITUDINAL if axis == 0 else GroundLineFamily.TRANSVERSE
                constraint = GroundDirectionConstraint(
                    node_id,
                    family,
                    line.points[0],
                    line.points[-1],
                    4.0,
                )
                direction_constraints.append(constraint)
                manual_direction_constraints.append(constraint)
    solution = solve_global_frame_graph(
        tuple(ordered),
        consistent_edges,
        reference_id,
        _pruning_rounds=0,
        direction_constraints=tuple(direction_constraints),
        reference_ground_to_image=reference_ground_to_image,
        absolute_ground_constraints=tuple(absolute_ground_constraints),
        absolute_ground_point_constraints=tuple(absolute_ground_point_constraints),
    )
    reference_frame = frames[reference_id]
    full_pitch_model = create_standard_full_pitch_marking_model()
    observations = []
    circle_observations = []
    contributing_nodes = 0
    for node in ordered:
        node_to_reference = solution.node_to_reference.get(node.node_id)
        if node_to_reference is None:
            continue
        reference_to_node = np.linalg.inv(node_to_reference)
        ground_to_node = reference_to_node @ reference_ground_to_image
        detection = detect_metric_ground_lines(
            frames[node.node_id], profile, ground_to_node, args.minimum_length
        )
        circle = detect_metric_center_circle(
            frames[node.node_id], profile, ground_to_node, full_pitch_model.center_circle_radius_m
        )
        if circle is not None:
            circle_observations.append(circle)
        if detection.lines:
            contributing_nodes += 1
        for line in detection.lines:
            observations.append(
                transform_line_observation(
                    ImageLineObservation(
                        line.family,
                        line.image_start,
                        line.image_end,
                        max(line.metric_length * line.confidence, 0.1),
                        float(np.mean((line.ground_start, line.ground_end), axis=0)[1 if line.family.value == "longitudinal" else 0]),
                        node.node_id,
                    ),
                    node_to_reference,
                )
            )
    for constraint in manual_direction_constraints:
        node_to_reference = solution.node_to_reference.get(constraint.node_id)
        if node_to_reference is None:
            continue
        observations.append(
            transform_line_observation(
                ImageLineObservation(
                    constraint.family,
                    constraint.image_start,
                    constraint.image_end,
                    25.0,
                    None,
                    f"manual-{constraint.node_id}",
                ),
                node_to_reference,
            )
        )
    orientation = None
    failure = None
    line_clusters = cluster_physical_lines(tuple(observations))
    line_diversity = summarize_line_diversity(line_clusters)
    marking_matches = {
        family.value: match_marking_offsets(
            tuple(
                cluster.mean_ground_offset_m
                for cluster in line_clusters[family]
                if cluster.mean_ground_offset_m is not None
            ),
            family,
            full_pitch_model,
        )
        for family in line_clusters
    }
    goal_zone_match = match_goal_zone_depth_lines(
        tuple(
            cluster.mean_ground_offset_m
            for cluster in line_clusters[GroundLineFamily.TRANSVERSE]
            if cluster.mean_ground_offset_m is not None
        ),
        create_goal_zone_reference("unknown"),
    )
    circle_consensus = estimate_circle_consensus(tuple(circle_observations))
    if circle_consensus is not None:
        consensus_evidence = GroundCircleEvidence(
            circle_consensus.ground_center,
            full_pitch_model.center_circle_radius_m,
            1.0,
            1.0,
            circle_consensus.confidence,
        )
        transverse_match = marking_matches["transverse"]
        if (
            not validate_ground_circle_on_frame(
                consensus_evidence,
                reference_frame,
                reference_ground_to_image,
            )
            or not circle_center_matches_halfway_line(
                circle_consensus.ground_center[0],
                transverse_match,
                full_pitch_model,
            )
        ):
            circle_consensus = None
    maximum_graph_error = 30.0
    if solution.edge_rms_px > 5.0 or solution.maximum_edge_error_px > maximum_graph_error:
        failure = (
            f"Framegraph-QA faalt: RMS {solution.edge_rms_px:.1f}px, "
            f"max {solution.maximum_edge_error_px:.1f}px."
        )
    else:
        try:
            orientation = estimate_orthogonal_ground_orientation(
                tuple(observations),
                (reference_frame.shape[1], reference_frame.shape[0]),
                minimum_lines_per_family=2 if args.format == "8v8" else 3,
            )
        except ValueError as error:
            failure = str(error)
    preview = reference_frame.copy()
    colors = {"longitudinal": (255, 255, 0), "transverse": (255, 0, 255)}
    for family_clusters in line_clusters.values():
        for index, cluster in enumerate(family_clusters, start=1):
            item = cluster.representative
            start = tuple(np.round(item.start).astype(int))
            end = tuple(np.round(item.end).astype(int))
            color = colors[item.family.value]
            cv2.line(preview, start, end, color, 3, cv2.LINE_AA)
            midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            cv2.putText(
                preview,
                f"{item.family.value[0].upper()}{index} ({cluster.observation_count}x)",
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )
    if circle_consensus is not None:
        circle = GroundCircleEvidence(
            circle_consensus.ground_center,
            full_pitch_model.center_circle_radius_m,
            1.0,
            1.0,
            circle_consensus.confidence,
        )
        polygon = project_ground_circle(circle, reference_ground_to_image)
        finite = np.all(np.isfinite(polygon), axis=1)
        if np.count_nonzero(finite) >= 3:
            cv2.polylines(
                preview,
                [np.round(polygon[finite]).astype(np.int32)],
                True,
                (0, 255, 255),
                4,
                cv2.LINE_AA,
            )
    status = "OPGELOST" if orientation is not None else f"ONVOLDOENDE: {failure}"
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 78), (20, 20, 20), -1)
    cv2.putText(preview, f"GLOBAL FRAMEGRAPH | nodes {len(solution.connected_nodes)}/{len(nodes)} | edges gebruikt {solution.used_edges}/{len(edges)} | {status}", (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 255), 2, cv2.LINE_AA)
    anchor_suffix = "" if args.reference_anchor == "goal-a" else f"_{args.reference_anchor}"
    preview_path = output_dir / f"{prefix}_global_frame_graph_ground{anchor_suffix}_qa.jpg"
    report_path = output_dir / f"{prefix}_global_frame_graph_ground{anchor_suffix}_qa.json"
    cv2.imwrite(str(preview_path), preview)
    report = {
        "schema_version": 1,
        "reference_anchor": args.reference_anchor,
        "nodes": len(nodes),
        "connected_nodes": len(solution.connected_nodes),
        "rejected_nodes": list(solution.rejected_nodes),
        "attempted_edges": attempted,
        "accepted_edges": len(edges),
        "tree_edges": len(tree_edges),
        "cycle_consistent_edges": len(consistent_edges),
        "direction_constraints": len(direction_constraints),
        "used_edges": solution.used_edges,
        "edge_rms_px": solution.edge_rms_px,
        "maximum_edge_error_px": solution.maximum_edge_error_px,
        "contributing_line_nodes": contributing_nodes,
        "line_observations": len(observations),
        "line_diversity": line_diversity,
        "full_pitch_reference": {
            "pitch_length_m": full_pitch_model.pitch_length_m,
            "pitch_width_m": full_pitch_model.pitch_width_m,
            "center_circle_radius_m": full_pitch_model.center_circle_radius_m,
            "penalty_area_depth_m": full_pitch_model.penalty_area_depth_m,
            "goal_area_depth_m": full_pitch_model.goal_area_depth_m,
        },
        "marking_matches": {
            family: result.to_dict() for family, result in marking_matches.items()
        },
        "reference_semantics": {
            "camera_anchor": reference_anchor.anchor_id,
            "eight_v_eight_goal_id": reference_anchor.goal_id,
            "eight_v_eight_goal_width_m": profile.goal_width_m,
            "eight_v_eight_goal_height_m": profile.goal_height_m,
            "full_pitch_goal_width_m": 7.32,
            "full_pitch_goal_height_m": 2.44,
            "full_pitch_goal_side": "unknown",
            "painted_white_lines_belong_to": "11v11_pitch",
            "eight_v_eight_boundary_reuse_requires_confirmation": True,
        },
        "full_pitch_goal_zone_match": goal_zone_match.to_dict(),
        "center_circle_observations": len(circle_observations),
        "center_circle_consensus": circle_consensus.to_dict() if circle_consensus is not None else None,
        "solved": orientation is not None,
        "failure_reason": failure,
        "orientation": orientation.to_dict() if orientation is not None else None,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    transverse_resolved = marking_matches[GroundLineFamily.TRANSVERSE.value].resolved
    longitudinal_available = line_diversity[GroundLineFamily.LONGITUDINAL.value]["cluster_count"] >= 1
    connected_ratio = len(solution.connected_nodes) / max(len(nodes), 1)
    playable_solved = (
        args.format == "8v8"
        and orientation is not None
        and connected_ratio >= 0.90
        and solution.edge_rms_px <= 8.0
        and solution.maximum_edge_error_px <= maximum_graph_error
        and (
            (transverse_resolved and longitudinal_available)
            or manual_metric_reference_applied
        )
    )
    registration_reason = (
        (
            "8v8-grondvlak opgelost met drie metrische grondankers en twee handmatig bevestigde "
            "globale verdwijnrichtingen."
            if manual_metric_reference_applied
            else "8v8-grondvlak opgelost met bevestigde gedeelde achterlijn en uniek 11v11-dwarslijnenpatroon."
        )
        if playable_solved
        else failure or "Onvoldoende globale lijn- of framedekking."
    )
    registered_frames = []
    for node in ordered:
        node_to_reference = solution.node_to_reference.get(node.node_id)
        if node_to_reference is None:
            continue
        reference_to_node = np.linalg.inv(node_to_reference)
        registered_frames.append(
            RegisteredGroundFrame(
                node.frame_number,
                node.time_seconds,
                reference_to_node @ reference_ground_to_image,
            )
        )
    registration = GlobalGroundRegistration(
        video.name,
        args.format,
        reference_anchor.anchor_id,
        tuple(registered_frames),
        connected_ratio,
        len(observations),
        playable_solved,
        registration_reason,
    )
    registration_path = output_dir / f"{prefix}_global_ground_registration.json"
    save_global_ground_registration(registration, registration_path)
    print(f"Nodes verbonden: {len(solution.connected_nodes)}/{len(nodes)}")
    print(f"Edges lokaal geldig: {len(edges)}/{attempted} | globaal gebruikt: {solution.used_edges} | RMS {solution.edge_rms_px:.2f}px | max {solution.maximum_edge_error_px:.2f}px")
    print(f"Lijnframes: {contributing_nodes} | lijnobservaties: {len(observations)} | {status}")
    for family, diagnostics in line_diversity.items():
        print(
            f"{family}: {diagnostics['cluster_count']} fysieke lijnen | "
            f"spreiding {diagnostics['metric_offset_span_m']:.1f}m / "
            f"{diagnostics['image_separation_px']:.0f}px"
        )
        match = marking_matches[family]
        print(f"  11v11-model: {'UNIEK' if match.resolved else 'NOG AMBIGU'} | {match.reason}")
        if match.hypotheses:
            print(f"  Beste kandidaat: {', '.join(match.hypotheses[0].marking_ids)}")
    if circle_consensus is None:
        print(f"Middencirkel: geen consensus ({len(circle_observations)} losse kandidaat/kandidaten)")
    else:
        print(
            f"Middencirkel: CONSENSUS | {circle_consensus.observations} frames | "
            f"RMS {circle_consensus.rms_m:.2f}m"
        )
    print(
        f"11v11-doelzonepatroon bij 8v8-doelanker {reference_anchor.goal_id}: "
        f"{'UNIEK' if goal_zone_match.resolved else 'ONVOLDOENDE'} | {goal_zone_match.reason}"
    )
    if goal_zone_match.marking_ids:
        print(
            f"  Lijnen: {', '.join(goal_zone_match.marking_ids)} | "
            f"schaal {goal_zone_match.scale:.3f} | RMS {goal_zone_match.rms_m:.2f}m"
        )
    print(f"QA-preview: {preview_path}")
    print(f"QA-rapport: {report_path}")
    print(
        f"Globale grondregistratie: {'OPGELOST' if playable_solved else 'ONVOLDOENDE'} | "
        f"{registration_path}"
    )


if __name__ == "__main__":
    main()
