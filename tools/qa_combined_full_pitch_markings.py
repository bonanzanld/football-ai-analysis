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

from football_ai.calibration.full_pitch_markings import (
    create_standard_full_pitch_marking_model,
    match_marking_offsets,
)
from football_ai.calibration.ground_line_evidence import GroundLineFamily


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combineer fysieke 11v11-lijnmarkeringen uit de doel-A- en doel-B-framegraphs."
    )
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{Path(args.video).stem}_{args.format}"
    reports = {
        "goal-a": output_dir / f"{prefix}_global_frame_graph_ground_qa.json",
        "goal-b": output_dir / f"{prefix}_global_frame_graph_ground_goal-b_qa.json",
    }
    loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in reports.items()}
    model = create_standard_full_pitch_marking_model()
    combined_offsets: dict[str, list[float]] = {}
    matches = {}
    for family in GroundLineFamily:
        values = []
        for report in loaded.values():
            values.extend(
                float(cluster["mean_ground_offset_m"])
                for cluster in report["line_diversity"][family.value]["clusters"]
                if cluster["mean_ground_offset_m"] is not None
            )
        merged = _merge_offsets(values)
        combined_offsets[family.value] = merged
        matches[family.value] = match_marking_offsets(tuple(merged), family, model)

    solved_families = sum(result.resolved for result in matches.values())
    status = "DEELS OPGELOST" if solved_families == 1 else "OPGELOST" if solved_families == 2 else "ONVOLDOENDE"
    report = {
        "schema_version": 1,
        "source_reports": {name: str(path) for name, path in reports.items()},
        "full_pitch_reference": {
            "pitch_length_m": model.pitch_length_m,
            "pitch_width_m": model.pitch_width_m,
            "center_circle_radius_m": model.center_circle_radius_m,
            "penalty_area_depth_m": model.penalty_area_depth_m,
            "goal_area_depth_m": model.goal_area_depth_m,
        },
        "combined_offsets_m": combined_offsets,
        "matches": {family: result.to_dict() for family, result in matches.items()},
        "center_circle_consensus": {
            anchor: source.get("center_circle_consensus") for anchor, source in loaded.items()
        },
        "full_pitch_goal_zone_matches": {
            anchor: source.get("full_pitch_goal_zone_match") for anchor, source in loaded.items()
        },
        "status": status,
    }
    report_path = output_dir / f"{prefix}_combined_full_pitch_markings_qa.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    previews = [
        cv2.imread(str(output_dir / f"{prefix}_global_frame_graph_ground_qa.jpg")),
        cv2.imread(str(output_dir / f"{prefix}_global_frame_graph_ground_goal-b_qa.jpg")),
    ]
    if all(item is not None for item in previews):
        height = min(item.shape[0] for item in previews)
        resized = [cv2.resize(item, (round(item.shape[1] * height / item.shape[0]), height)) for item in previews]
        canvas = np.hstack(resized)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 84), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            f"GECOMBINEERD 11v11-LIJNMODEL | {status}",
            (18, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        summary = " | ".join(
            f"{family}: {len(combined_offsets[family])} lijnen, {'uniek' if matches[family].resolved else 'ambigu'}"
            for family in combined_offsets
        )
        cv2.putText(canvas, summary, (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
        preview_path = output_dir / f"{prefix}_combined_full_pitch_markings_qa.jpg"
        cv2.imwrite(str(preview_path), canvas)
        print(f"QA-preview: {preview_path}")

    for family, result in matches.items():
        print(f"{family}: {len(combined_offsets[family])} fysieke lijnen | {'UNIEK' if result.resolved else 'AMBIGU'}")
        if result.hypotheses:
            best = result.hypotheses[0]
            print(f"  Beste: {', '.join(best.marking_ids)} | schaal {best.scale:.3f} | RMS {best.rms_m:.2f}m")
    for anchor, source in loaded.items():
        goal_zone = source.get("full_pitch_goal_zone_match")
        if goal_zone is None:
            continue
        print(
            f"11v11-doelzonepatroon bij {anchor}: "
            f"{'UNIEK' if goal_zone['resolved'] else 'ONVOLDOENDE'} | {goal_zone['reason']}"
        )
    print(f"Status: {status}")
    print(f"QA-rapport: {report_path}")


def _merge_offsets(values: list[float], threshold_m: float = 1.5) -> list[float]:
    groups: list[list[float]] = []
    for value in sorted(values):
        for group in groups:
            if abs(value - float(np.mean(group))) < threshold_m:
                group.append(value)
                break
        else:
            groups.append([value])
    return [float(np.mean(group)) for group in groups]


if __name__ == "__main__":
    main()
