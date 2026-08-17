from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.calibration.lens_intrinsics_io import load_lens_intrinsics
from football_ai.calibration.local_field_atlas import load_local_field_atlas
from football_ai.calibration.global_frame_graph import (
    FrameGraphNode,
    estimate_frame_edge,
    estimate_ground_frame_edge,
    select_cycle_consistent_edges,
    select_maximum_quality_tree,
    solve_global_frame_graph,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vergelijk lokale atlasvlakken in hun overlapzone.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    args = parser.parse_args()

    video = PROJECT_ROOT / "videos" / args.video
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    atlas = load_local_field_atlas(output / f"{prefix}_local_field_atlas.json")
    lens, lens_source = load_lens_intrinsics(
        output / f"{prefix}_lens_geometry_qa.json",
        selected_zoom_path=output / f"{prefix}_selected_fixed_zoom_segment.json",
    )
    patches = {patch.patch_id: patch for patch in atlas.patches}
    required = {"goal-a", "goal-b"}
    if not required <= patches.keys():
        raise RuntimeError("Overlap-QA vereist zowel goal-a als goal-b.")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    first = min(patches[item].anchor_frame for item in required)
    last = max(patches[item].anchor_frame for item in required)
    transforms, graph_diagnostics = _connect_anchors(
        capture, lens, first, last, patches["goal-b"].anchor_frame, fps
    )
    target = patches["goal-b"].anchor_frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, raw = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Controleframe kon niet worden gelezen.")
    frame = cv2.undistort(raw, lens.camera_matrix, lens.distortion_coefficients)
    minimum_x = max(min(point[0] for point in patches[name].support_polygon) for name in required)
    maximum_x = min(max(point[0] for point in patches[name].support_polygon) for name in required)
    ground = np.asarray(
        [
            (x, y)
            for x in np.linspace(minimum_x, maximum_x, 5)
            for y in np.linspace(0.0, atlas.pitch_width_m, 5)
        ],
        dtype=np.float64,
    )
    projected = {
        "goal-a": _project(
            transforms[f"f{patches['goal-a'].anchor_frame}"]
            @ patches["goal-a"].ground_to_anchor,
            ground,
        ),
        "goal-b": _project(patches["goal-b"].ground_to_anchor, ground),
    }
    errors = np.linalg.norm(projected["goal-a"] - projected["goal-b"], axis=1)
    for color, name in (((0, 220, 255), "goal-a"), ((255, 180, 0), "goal-b")):
        points = projected[name].reshape(5, 5, 2)
        for row in points:
            cv2.polylines(frame, [np.rint(row).astype(np.int32)], False, color, 3, cv2.LINE_AA)
        for column in points.transpose(1, 0, 2):
            cv2.polylines(frame, [np.rint(column).astype(np.int32)], False, color, 3, cv2.LINE_AA)
    median = float(np.median(errors))
    maximum = float(np.max(errors))
    median_limit = 0.035 * frame.shape[0]
    maximum_limit = 0.10 * frame.shape[0]
    accepted = median <= median_limit and maximum <= maximum_limit
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 92), (15, 15, 15), -1)
    cv2.putText(
        frame, f"OVERLAP GOAL A/B | {target / fps:.1f}s", (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"mediaan {median:.1f}px | maximum {maximum:.1f}px | geel=A blauw=B",
        (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA,
    )
    preview = output / f"{prefix}_local_field_atlas_overlap_qa.jpg"
    cv2.imwrite(str(preview), frame)
    report = {
        "schema_version": 1,
        "video": video.name,
        "target_frame": target,
        "target_seconds": target / fps,
        "overlap_x_metres": [minimum_x, maximum_x],
        "sample_points": len(ground),
        "median_disagreement_px": median,
        "maximum_disagreement_px": maximum,
        "acceptance": {
            "accepted": accepted,
            "median_limit_px": median_limit,
            "maximum_limit_px": maximum_limit,
        },
        "per_point_disagreement_px": errors.tolist(),
        "frame_graph": graph_diagnostics,
        "lens_source": lens_source,
    }
    report_path = output / f"{prefix}_local_field_atlas_overlap_qa.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    verdict = "GEACCEPTEERD" if accepted else "AFGEKEURD"
    print(f"Overlap A/B: {verdict} | mediaan {median:.1f}px | maximum {maximum:.1f}px")
    print(f"Preview: {preview}")
    print(f"Rapport: {report_path}")


def _connect_anchors(capture, lens, first, last, reference_frame, fps):
    step = max(1, int(round(0.5 * fps)))
    frame_numbers = sorted({first, last, reference_frame, *range(first, last + 1, step)})
    nodes = []
    frames = {}
    for frame_number in frame_numbers:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            continue
        corrected = cv2.undistort(frame, lens.camera_matrix, lens.distortion_coefficients)
        node_id = f"f{frame_number}"
        nodes.append(FrameGraphNode(node_id, frame_number, frame_number / fps))
        frames[node_id] = corrected
    edges = []
    for index, source in enumerate(nodes):
        for gap in (1, 2):
            if index + gap >= len(nodes):
                continue
            target = nodes[index + gap]
            try:
                try:
                    edge = estimate_ground_frame_edge(
                        source.node_id, target.node_id,
                        frames[source.node_id], frames[target.node_id],
                    )
                except ValueError:
                    edge = estimate_frame_edge(
                        source.node_id, target.node_id,
                        frames[source.node_id], frames[target.node_id],
                    )
                edges.append(edge)
            except ValueError:
                pass
    reference_id = f"f{reference_frame}"
    tree = select_maximum_quality_tree(tuple(nodes), tuple(edges))
    initial = solve_global_frame_graph(tuple(nodes), tree, reference_id, _pruning_rounds=0)
    consistent = select_cycle_consistent_edges(initial, tree, tuple(edges), maximum_error_px=10.0)
    solution = solve_global_frame_graph(
        tuple(nodes), consistent, reference_id, _pruning_rounds=2
    )
    required = {f"f{first}", f"f{last}"}
    missing = required - solution.node_to_reference.keys()
    if missing:
        raise RuntimeError(f"Framegraph verbindt beide doelankers niet: {sorted(missing)}")
    diagnostics = {
        "nodes": len(nodes),
        "candidate_edges": len(edges),
        "consistent_edges": len(consistent),
        "connected_nodes": len(solution.connected_nodes),
        "edge_rms_px": solution.edge_rms_px,
        "maximum_edge_error_px": solution.maximum_edge_error_px,
        "sample_interval_seconds": 0.5,
    }
    return solution.node_to_reference, diagnostics


def _project(matrix, ground):
    homogeneous = np.column_stack((ground, np.ones(len(ground))))
    projected = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T
    if np.any(np.abs(projected[:, 2]) < 1e-9):
        raise RuntimeError("Overlapprojectie raakt het verdwijnvlak.")
    return projected[:, :2] / projected[:, 2:3]


if __name__ == "__main__":
    main()
