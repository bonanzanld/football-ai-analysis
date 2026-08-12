from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.player_projection_evidence import evaluate_player_footpoints
from football_ai.calibration.video_projection_plan import (
    gate_projection_plan_with_player_evidence,
    load_video_projection_plan,
    save_video_projection_plan,
)
from football_ai.detector import FootballDetector
from football_ai.filtering.player_filter import PlayerFilter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use detected player footpoints as diagnostic projection-plan evidence."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gated-plan-output",
        type=Path,
        help="Optioneel uitvoerplan waarin door spelers verworpen projecties zijn verwijderd; vereist stride 1.",
    )
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--player-threshold", type=float, default=0.20)
    parser.add_argument(
        "--minimum-player-containment",
        type=float,
        default=0.60,
        help="Minimum aandeel voetpunten binnen het veld of de zachte grensmarge.",
    )
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    if args.gated_plan_output is not None and args.stride != 1:
        parser.error("--gated-plan-output requires --stride 1")
    if not 0.5 <= args.minimum_player_containment <= 1.0:
        parser.error("--minimum-player-containment must be between 0.5 and 1.0")

    video_path = args.video if args.video.is_absolute() else PROJECT_ROOT / args.video
    plan_path = args.plan if args.plan.is_absolute() else PROJECT_ROOT / args.plan
    plan = load_video_projection_plan(plan_path)
    profile = create_detection_profile(plan.match_format)
    detector = FootballDetector(player_threshold=args.player_threshold, ball_threshold=0.05)
    player_filter = PlayerFilter(
        minimum_box_height=24,
        minimum_aspect_ratio=1.15,
        maximum_aspect_ratio=6.0,
        minimum_foot_y_ratio=0.15,
        minimum_green_ratio=0.18,
        pitch_calibration=None,
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    records = []
    try:
        sampled = plan.records[:: args.stride]
        for index, planned in enumerate(sampled, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, planned.frame_number)
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"Cannot read frame {planned.frame_number}")
            _all, people, _balls = detector.detect(frame)
            people = player_filter.filter(
                frame=frame,
                detections=people,
                frame_number=planned.frame_number,
            )
            footpoints = tuple(
                ((float(box[0]) + float(box[2])) / 2.0, float(box[3]))
                for box in people.xyxy
            )
            split_x = frame.shape[1] / 2.0
            evidence = evaluate_player_footpoints(
                planned.projection,
                footpoints,
                pitch_length_m=profile.pitch_length_m,
                pitch_width_m=profile.pitch_width_m,
                tolerated_outside_m=profile.boundary_layout_tolerance_m,
                severe_outside_m=max(8.0, 2.5 * profile.boundary_layout_tolerance_m),
                minimum_acceptable_ratio=args.minimum_player_containment,
            )
            left_evidence = evaluate_player_footpoints(
                planned.projection,
                (point for point in footpoints if point[0] < split_x),
                pitch_length_m=profile.pitch_length_m,
                pitch_width_m=profile.pitch_width_m,
                tolerated_outside_m=profile.boundary_layout_tolerance_m,
                severe_outside_m=max(8.0, 2.5 * profile.boundary_layout_tolerance_m),
                minimum_acceptable_ratio=args.minimum_player_containment,
            )
            right_evidence = evaluate_player_footpoints(
                planned.projection,
                (point for point in footpoints if point[0] >= split_x),
                pitch_length_m=profile.pitch_length_m,
                pitch_width_m=profile.pitch_width_m,
                tolerated_outside_m=profile.boundary_layout_tolerance_m,
                severe_outside_m=max(8.0, 2.5 * profile.boundary_layout_tolerance_m),
                minimum_acceptable_ratio=args.minimum_player_containment,
            )
            side_warning = (
                left_evidence.classification == "rejected"
                or right_evidence.classification == "rejected"
            )
            classification = evidence.classification
            records.append(
                {
                    "frame_number": planned.frame_number,
                    "time_seconds": planned.time_seconds,
                    "plan_status": planned.status,
                    "anchor_id": planned.anchor_id,
                    **asdict(evidence),
                    "global_classification": evidence.classification,
                    "classification": classification,
                    "side_warning": side_warning,
                    "left_half": asdict(left_evidence),
                    "right_half": asdict(right_evidence),
                }
            )
            print(
                f"{index}/{len(sampled)} frame {planned.frame_number}: "
                f"{len(footpoints)} players, {classification} "
                f"(L {left_evidence.classification}, R {right_evidence.classification})"
            )
    finally:
        capture.release()

    by_status = {}
    for status in sorted({item["plan_status"] for item in records}):
        selected = [item for item in records if item["plan_status"] == status]
        projected = sum(int(item["projected_count"]) for item in selected)
        by_status[status] = {
            "frames": len(selected),
            "players": projected,
            "frames_over_expected_16_players": sum(
                bool(item["exceeds_expected_player_count"]) for item in selected
            ),
            "inside_ratio": sum(int(item["inside_count"]) for item in selected) / max(1, projected),
            "acceptable_ratio": sum(
                int(item["inside_count"]) + int(item["tolerated_outside_count"])
                for item in selected
            ) / max(1, projected),
            "classifications": dict(Counter(str(item["classification"]) for item in selected)),
        }
    payload = {
        "schema_version": 1,
        "video": str(video_path),
        "projection_plan": str(plan_path),
        "diagnostic_only": True,
        "boundary_layout_tolerance_m": profile.boundary_layout_tolerance_m,
        "minimum_player_containment": args.minimum_player_containment,
        "stride": args.stride,
        "summary_by_plan_status": by_status,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Report: {args.output}")
    if args.gated_plan_output is not None:
        gated = gate_projection_plan_with_player_evidence(
            plan,
            {int(item["frame_number"]): str(item["classification"]) for item in records},
        )
        save_video_projection_plan(gated, args.gated_plan_output)
        rejected = sum(
            before.projection_matrix is not None and after.projection_matrix is None
            for before, after in zip(plan.records, gated.records)
        )
        print(f"Gated plan: {args.gated_plan_output} | rejected projections: {rejected}")


if __name__ == "__main__":
    main()
