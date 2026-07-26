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

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.camera_anchor_bank_3d import load_camera_anchor_bank
from football_ai.calibration.camera_anchor_bank_3d import (
    refine_camera_anchor_bank_ground,
    save_camera_anchor_bank,
)
from football_ai.calibration.camera_anchor_runtime import CameraAnchorRuntime
from football_ai.calibration.reference_3d import create_field_reference_3d
from football_ai.calibration.global_ground_registration import load_global_ground_registration
from football_ai.calibration.video_projection_plan import (
    OfflineVideoProjectionAnalyzer,
    save_video_projection_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyseer eerst de camerabeweging en bouw daarna een vast projectieplan."
    )
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--start", type=float, default=900.0)
    parser.add_argument("--duration", type=float, default=110.0)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()

    video_path = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video_path.stem}_{args.format}"
    expanded = output_dir / f"{prefix}_camera_anchors_3d_expanded.json"
    primary = output_dir / f"{prefix}_camera_anchors_3d.json"
    bank = load_camera_anchor_bank(expanded if expanded.exists() else primary)
    contour_report_path = output_dir / f"{prefix}_playable_field_contour_qa.json"
    contour_report = json.loads(contour_report_path.read_text(encoding="utf-8"))
    bank = refine_camera_anchor_bank_ground(bank, contour_report)
    refined_bank_path = output_dir / f"{prefix}_camera_anchors_3d_field_refined.json"
    save_camera_anchor_bank(bank, refined_bank_path)
    quality = _load_static_quality(contour_report_path)
    trusted = {
        anchor.anchor_id
        for anchor in bank.anchors
        if quality.get(anchor.parent_anchor_id or anchor.anchor_id, False)
    }
    profile = create_detection_profile(args.format)
    reference = create_field_reference_3d(profile)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Video kon niet worden geopend: {video_path}")
    anchor_frames = {}
    for anchor in bank.anchors:
        capture.set(cv2.CAP_PROP_POS_FRAMES, anchor.frame_number)
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Ankerframe {anchor.frame_number} kon niet worden gelezen.")
        anchor_frames[anchor.anchor_id] = frame
    runtime = CameraAnchorRuntime(bank, reference, anchor_frames)
    global_path = output_dir / f"{prefix}_global_ground_registration.json"
    global_registration = (
        load_global_ground_registration(global_path) if global_path.exists() else None
    )
    analyzer = OfflineVideoProjectionAnalyzer(
        runtime,
        profile,
        trusted,
        global_registration=global_registration,
    )
    print("Fase 1/2: volledige videoperiode analyseren; er wordt nog geen QA-video gemaakt.")
    print(f"Gezamenlijk veldmodel toegepast: {refined_bank_path}")
    if global_registration is not None:
        print(
            f"Globale grondregistratie: "
            f"{'OPGELOST' if global_registration.solved_for_playable_field else 'ONVOLDOENDE'} | "
            f"{len(global_registration.frames)} camerastanden"
        )
    plan = analyzer.analyze(
        capture,
        str(video_path),
        args.format,
        args.start,
        args.duration,
        args.interval,
    )
    capture.release()
    output = output_dir / f"{prefix}_projection_plan.json"
    save_video_projection_plan(plan, output)
    resolved = sum(item.projection_matrix is not None for item in plan.records)
    trusted_count = sum(item.status == "valid" for item in plan.records)
    candidate_count = sum(item.status == "candidate" for item in plan.records)
    unknown = len(plan.records) - resolved
    print(f"Projectieplan opgeslagen: {output}")
    print(
        f"Vooranalyse: betrouwbaar {trusted_count} | kandidaat {candidate_count} | "
        f"onopgelost {unknown} | totaal {len(plan.records)}"
    )
    if plan.resolved_ratio < 0.80:
        print(
            "VOORANALYSE ONVOLDOENDE: minder dan 80% is geometrisch opgelost. "
            "De QA-video mag dit plan alleen diagnostisch weergeven."
        )
    elif plan.trusted_ratio < 0.80:
        print(
            "VOORANALYSE WAARSCHUWING: voldoende projecties gevonden, maar minder dan 80% "
            "komt uit statisch goedgekeurde ankers."
        )
    else:
        print("VOORANALYSE GESLAAGD: het plan kan voor de QA-video worden gebruikt.")


def _load_static_quality(path: Path) -> dict[str, bool]:
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for anchor_id in report.get("projection_quality", {}):
        result[anchor_id] = all(
            section.get(anchor_id, {}).get("confirmed", section.get(anchor_id, {}).get("valid", False))
            for section in (
                report.get("projection_quality", {}),
                report.get("parallelism_quality", {}),
                report.get("support_alignment_quality", {}),
                report.get("orthogonality_quality", {}),
            )
        )
    return result


if __name__ == "__main__":
    main()
