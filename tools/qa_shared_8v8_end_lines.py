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
from football_ai.calibration.bootstrap.white_line_detection import detect_white_field_lines
from football_ai.calibration.shared_end_line_binding import assess_shared_full_pitch_sideline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bevestig dat 8v8-achterlijnen bestaande witte 11v11-zijlijnen hergebruiken."
    )
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4")
    parser.add_argument("--format", choices=("8v8",), default="8v8")
    parser.add_argument(
        "--confirm-shared-sidelines",
        action="store_true",
        help="Bevestig expliciet dat beide 8v8-achterlijnen de twee witte 11v11-zijlijnen hergebruiken.",
    )
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    seeds = load_goal_seeds(output_dir / f"{prefix}_goal_seeds.json")
    profile = create_detection_profile(args.format)
    capture = cv2.VideoCapture(str(video))
    previews, assessments = [], []
    for seed in seeds:
        capture.set(cv2.CAP_PROP_POS_FRAMES, seed.frame_number)
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Frame {seed.frame_number} kon niet worden gelezen.")
        detection = detect_white_field_lines(frame, profile)
        assessment = assess_shared_full_pitch_sideline(
            seed,
            detection.candidates,
            profile.goal_width_m,
            operator_confirmed_layout=args.confirm_shared_sidelines,
        )
        assessments.append(assessment)
        preview = frame.copy()
        cv2.line(
            preview,
            tuple(np.round(seed.first_ground).astype(int)),
            tuple(np.round(seed.second_ground).astype(int)),
            (255, 0, 255),
            6,
            cv2.LINE_AA,
        )
        if assessment.matched_line is not None:
            candidate = assessment.matched_line
            cv2.line(
                preview,
                tuple(np.round(candidate.start).astype(int)),
                tuple(np.round(candidate.end).astype(int)),
                (0, 255, 255),
                5,
                cv2.LINE_AA,
            )
        status = "BEVESTIGD" if assessment.binding.confirmed else "NIET BEVESTIGD"
        if assessment.binding.confirmed and not assessment.visual_confirmation:
            status += " (VELDOPSTELLING)"
        color = (40, 220, 40) if assessment.binding.confirmed else (0, 80, 255)
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 64), (20, 20, 20), -1)
        cv2.putText(
            preview,
            f"8v8 DOEL {seed.goal_id} 5x2m | 11v11-ZIJLIJN ALS ACHTERLIJN: {status}",
            (14, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            color,
            2,
            cv2.LINE_AA,
        )
        previews.append(preview)
    capture.release()
    preview_path = output_dir / f"{prefix}_shared_end_lines_qa.jpg"
    cv2.imwrite(str(preview_path), np.hstack(previews))
    report_path = output_dir / f"{prefix}_shared_end_lines_qa.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "painted_white_lines_belong_to": "11v11_pitch",
                "eight_v_eight_goal_dimensions_m": [profile.goal_width_m, profile.goal_height_m],
                "assessments": [item.to_dict() for item in assessments],
                "all_end_lines_confirmed": all(item.binding.confirmed for item in assessments),
                "operator_confirmed_shared_sidelines": args.confirm_shared_sidelines,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for item in assessments:
        distance = "n.v.t." if item.maximum_post_distance_px is None else f"{item.maximum_post_distance_px:.1f}px"
        print(f"8v8-doel {item.goal_id}: {'BEVESTIGD' if item.binding.confirmed else 'NIET BEVESTIGD'} | paal-lijnafstand {distance}")
    print(f"QA-preview: {preview_path}")
    print(f"QA-rapport: {report_path}")


if __name__ == "__main__":
    main()
