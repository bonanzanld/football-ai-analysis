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

from football_ai.calibration.global_frame_graph import (
    FrameGraphNode,
    GroundDirectionConstraint,
    estimate_frame_edge,
    estimate_ground_frame_edge,
    select_cycle_consistent_edges,
    select_maximum_quality_tree,
    solve_global_frame_graph,
)
from football_ai.calibration.ground_line_evidence import GroundLineFamily
from football_ai.calibration.lens_geometry import LensIntrinsics
from football_ai.calibration.local_field_atlas import LocalFieldAtlas, load_local_field_atlas, save_local_field_atlas
from football_ai.calibration.manual_midfield_line import load_manual_midfield_line
from football_ai.calibration.midfield_atlas_anchor import (
    create_positioned_midfield_patch,
    midfield_direction_error_px,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Voeg een zelfstandig middenveldvlak toe aan de lokale atlas.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.15 <= args.interval <= 1.0:
        raise ValueError("Het framegraphinterval moet tussen 0,15 en 1,0 seconde liggen.")

    video = PROJECT_ROOT / "videos" / args.video
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    atlas_path = output / f"{prefix}_local_field_atlas.json"
    atlas = load_local_field_atlas(atlas_path)
    midfield = load_manual_midfield_line(output / f"{prefix}_manual_midfield_line.json")
    lens_data = json.loads((output / f"{prefix}_lens_geometry_qa.json").read_text())
    lens = LensIntrinsics(
        tuple(lens_data["frame_size"]), float(lens_data["focal_length_px"]),
        tuple(lens_data["principal_point"]), tuple(lens_data["radial_distortion"]),
    )
    goal_a_patch = next(item for item in atlas.patches if item.patch_id == "goal-a")
    reference_patch = next(item for item in atlas.patches if item.patch_id == "goal-b")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video kon niet worden geopend: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    position_frames = tuple(
        frame for frame in (
            midfield.rear_sideline_frame_number,
            midfield.front_sideline_frame_number,
        ) if frame is not None
    )
    required_frames = (
        midfield.frame_number, goal_a_patch.anchor_frame, reference_patch.anchor_frame,
        *position_frames,
    )
    start = min(required_frames)
    end = max(required_frames)
    step = max(1, int(round(args.interval * fps)))
    frame_numbers = list(range(start, end + 1, step))
    frame_numbers.extend(
        (
            start,
            midfield.frame_number,
            reference_patch.anchor_frame,
            goal_a_patch.anchor_frame,
            *position_frames,
            end,
        )
    )
    frame_numbers = sorted(set(frame_numbers))
    nodes, frames = [], {}
    for index, frame_number in enumerate(frame_numbers):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            continue
        node_id = f"f{frame_number}"
        nodes.append(FrameGraphNode(node_id, frame_number, frame_number / fps))
        frames[node_id] = cv2.undistort(frame, lens.camera_matrix, lens.distortion_coefficients)
        if index and index % 30 == 0:
            print(f"Middenframegraph: {index}/{len(frame_numbers)} beelden geladen")
    capture.release()
    ordered = tuple(nodes)
    edges = []
    for source_index, source in enumerate(ordered):
        for gap in (1, 2):
            target_index = source_index + gap
            if target_index >= len(ordered):
                continue
            target = ordered[target_index]
            try:
                try:
                    edge = estimate_ground_frame_edge(
                        source.node_id, target.node_id, frames[source.node_id], frames[target.node_id]
                    )
                except ValueError:
                    edge = estimate_frame_edge(
                        source.node_id, target.node_id, frames[source.node_id], frames[target.node_id]
                    )
                edges.append(edge)
            except ValueError:
                pass
        if source_index and source_index % 25 == 0:
            print(f"Middenframegraph: {source_index}/{len(ordered)} verbindingen verwerkt")

    reference_id = f"f{reference_patch.anchor_frame}"
    midfield_id = f"f{midfield.frame_number}"
    tree = select_maximum_quality_tree(ordered, tuple(edges))
    tree_solution = solve_global_frame_graph(ordered, tree, reference_id, _pruning_rounds=0)
    if midfield_id not in tree_solution.node_to_reference:
        raise RuntimeError(
            "Het middenframe is niet via betrouwbare overlappende grondbeelden met Doel B verbonden. "
            "Probeer opnieuw met --interval 0.25."
        )
    consistent = select_cycle_consistent_edges(tree_solution, tree, tuple(edges), maximum_error_px=10.0)
    corrected_points = lens.undistort_points(np.asarray(midfield.points, dtype=np.float64))
    center = np.mean(corrected_points, axis=0)
    _u, _s, vh = np.linalg.svd(corrected_points - center)
    direction = vh[0]
    first, second = center - 800.0 * direction, center + 800.0 * direction
    corrected_line = np.cross(
        np.asarray((first[0], first[1], 1.0), dtype=np.float64),
        np.asarray((second[0], second[1], 1.0), dtype=np.float64),
    )
    corrected_line /= np.linalg.norm(corrected_line[:2])
    if midfield.rear_sideline_point is None or midfield.front_sideline_point is None:
        raise RuntimeError(
            "De middenreferentie mist een van de twee zichtbare 8v8-zijlijnposities. Voer opnieuw "
            "tools/collect_manual_midfield_line.py uit; klik na de vijf witte-lijnpunten "
            "ACHTER en blader zo nodig naar een ander videomoment voor VOOR."
        )
    constraint = GroundDirectionConstraint(
        midfield_id, GroundLineFamily.LONGITUDINAL, tuple(first), tuple(second), 25.0
    )
    solution = solve_global_frame_graph(
        ordered, consistent, reference_id, _pruning_rounds=2,
        direction_constraints=(constraint,),
        reference_ground_to_image=reference_patch.ground_to_anchor,
    )
    if midfield_id not in solution.node_to_reference:
        raise RuntimeError("Het middenanker viel weg tijdens de globale consistentiecontrole.")
    goal_a_id = f"f{goal_a_patch.anchor_frame}"
    if goal_a_id not in solution.node_to_reference:
        raise RuntimeError("Doel A viel weg tijdens de globale consistentiecontrole.")
    midfield_to_goal_b = solution.node_to_reference[midfield_id]
    goal_a_to_goal_b = solution.node_to_reference[goal_a_id]
    midfield_to_goal_a = np.linalg.inv(goal_a_to_goal_b) @ midfield_to_goal_b
    ground_to_midfield_from_goal_a = np.linalg.inv(midfield_to_goal_a) @ goal_a_patch.ground_to_anchor
    ground_to_midfield_from_goal_b = np.linalg.inv(midfield_to_goal_b) @ reference_patch.ground_to_anchor
    def position_in_midfield(point, frame_number):
        corrected = lens.undistort_points(np.asarray((point,), dtype=np.float64))[0]
        source_frame = midfield.frame_number if frame_number is None else int(frame_number)
        if source_frame == midfield.frame_number:
            return tuple(map(float, corrected))
        source_id = f"f{source_frame}"
        if source_id not in solution.node_to_reference:
            raise RuntimeError(
                f"Zijlijnpositie uit frame {source_frame} kon niet met het middenbeeld worden gekoppeld."
            )
        source_to_midfield = np.linalg.inv(midfield_to_goal_b) @ solution.node_to_reference[source_id]
        homogeneous = source_to_midfield @ np.asarray((*corrected, 1.0), dtype=np.float64)
        if abs(float(homogeneous[2])) < 1e-9:
            raise RuntimeError("Een zijlijnpositie projecteert naar oneindig in het middenbeeld.")
        return tuple(map(float, homogeneous[:2] / homogeneous[2]))

    corrected_rear = position_in_midfield(
        midfield.rear_sideline_point, midfield.rear_sideline_frame_number
    )
    rear_patch = create_positioned_midfield_patch(
        midfield.frame_number,
        ground_to_midfield_from_goal_a,
        ground_to_midfield_from_goal_b,
        corrected_line,
        corrected_rear,
        None,
        atlas.pitch_length_m, atlas.pitch_width_m,
        solution.edge_rms_px, solution.maximum_edge_error_px,
        "midfield-rear",
    )
    line_error = midfield_direction_error_px(rear_patch, corrected_line)
    if line_error > 12.0:
        raise RuntimeError(f"Middenanker volgt de handmatige middenlijn onvoldoende: {line_error:.1f}px.")

    front_frame = int(midfield.front_sideline_frame_number)
    front_id = f"f{front_frame}"
    front_to_goal_b = solution.node_to_reference[front_id]
    front_to_goal_a = np.linalg.inv(goal_a_to_goal_b) @ front_to_goal_b
    ground_to_front_from_goal_a = np.linalg.inv(front_to_goal_a) @ goal_a_patch.ground_to_anchor
    ground_to_front_from_goal_b = np.linalg.inv(front_to_goal_b) @ reference_patch.ground_to_anchor
    midfield_to_front = np.linalg.inv(front_to_goal_b) @ midfield_to_goal_b
    front_line = np.linalg.inv(midfield_to_front).T @ corrected_line
    front_line /= np.linalg.norm(front_line[:2])
    corrected_front = tuple(map(float, lens.undistort_points(
        np.asarray((midfield.front_sideline_point,), dtype=np.float64)
    )[0]))
    front_patch = create_positioned_midfield_patch(
        front_frame,
        # This is deliberately a local patch.  Goal B is the nearer, internally
        # consistent metric basis for this view; importing the remote goal-A
        # end line here would reintroduce accumulated long-chain drift.
        ground_to_front_from_goal_b,
        ground_to_front_from_goal_b,
        front_line,
        None,
        corrected_front,
        atlas.pitch_length_m, atlas.pitch_width_m,
        solution.edge_rms_px, solution.maximum_edge_error_px,
        "midfield-front",
    )
    front_line_error = midfield_direction_error_px(front_patch, front_line)
    if front_line_error > 12.0:
        raise RuntimeError(
            f"Voorste middenanker volgt de lijnrichting onvoldoende: {front_line_error:.1f}px."
        )
    patches = tuple(
        item for item in atlas.patches if not item.patch_id.startswith("midfield")
    ) + (rear_patch, front_patch)
    updated = LocalFieldAtlas(
        atlas.video_name, atlas.match_format, atlas.pitch_length_m, atlas.pitch_width_m,
        patches, atlas.manual_midfield_line, atlas.manual_parallel_lines,
    )
    save_local_field_atlas(updated, atlas_path)
    report = {
        "schema_version": 1,
        "video": video.name,
        "reference_patch": reference_patch.patch_id,
        "midfield_frame": midfield.frame_number,
        "sample_interval_seconds": args.interval,
        "nodes": len(ordered),
        "candidate_edges": len(edges),
        "consistent_edges": len(consistent),
        "connected_nodes": len(solution.connected_nodes),
        "corner_sources": {"end_line_a": "goal-a", "end_line_b": "goal-b"},
        "sideline_position_sources": {
            "rear": {"patch": "midfield-rear", "frame": midfield.frame_number},
            "front": {"patch": "midfield-front", "frame": front_frame},
        },
        "graph_rms_px": solution.edge_rms_px,
        "maximum_graph_error_px": solution.maximum_edge_error_px,
        "midfield_line_error_px": line_error,
        "rear_confidence": rear_patch.confidence,
        "front_confidence": front_patch.confidence,
    }
    report_path = output / f"{prefix}_midfield_atlas_anchor.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Twee lokale middenveldankers toegevoegd: {atlas_path}")
    print(
        f"Frame {midfield.frame_number} ({midfield.time_seconds:.1f}s) | "
        f"graph RMS {solution.edge_rms_px:.2f}px | max {solution.maximum_edge_error_px:.2f}px | "
        f"lijnfout achter {line_error:.2f}px | lijnfout voor {front_line_error:.2f}px | "
        f"vertrouwen {rear_patch.confidence:.0%}/{front_patch.confidence:.0%}"
    )
    print(f"Diagnostiek: {report_path}")


if __name__ == "__main__":
    main()
