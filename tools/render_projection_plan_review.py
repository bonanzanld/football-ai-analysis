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

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.video_projection_plan import load_video_projection_plan


def _evenly_spaced(records: list[dict], count: int) -> list[dict]:
    if len(records) <= count:
        return records
    indices = np.linspace(0, len(records) - 1, count).round().astype(int)
    return [records[int(index)] for index in indices]


def _project(projection, points: list[tuple[float, float, float]]) -> np.ndarray:
    return np.round([projection.project(point) for point in points]).astype(np.int32)


def _draw_field(frame: np.ndarray, projection, length: float, width: float) -> None:
    boundary = _project(
        projection,
        [(0, 0, 0), (length, 0, 0), (length, width, 0), (0, width, 0)],
    )
    overlay = frame.copy()
    cv2.fillPoly(overlay, [boundary], (40, 190, 60), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.14, frame, 0.86, 0.0, frame)
    cv2.polylines(frame, [boundary], True, (0, 255, 255), 5, cv2.LINE_AA)
    midline = _project(projection, [(length / 2, 0, 0), (length / 2, width, 0)])
    cv2.line(frame, tuple(midline[0]), tuple(midline[1]), (255, 255, 255), 4, cv2.LINE_AA)
    center = _project(projection, [(length / 2, width / 2, 0)])[0]
    cv2.circle(frame, tuple(center), 7, (255, 255, 255), -1, cv2.LINE_AA)
    for x in (0.0, length):
        goal = _project(projection, [(x, width / 2 - 2.5, 0), (x, width / 2 + 2.5, 0)])
        cv2.line(frame, tuple(goal[0]), tuple(goal[1]), (255, 80, 255), 7, cv2.LINE_AA)
    for index, point in enumerate(boundary, start=1):
        cv2.circle(frame, tuple(point), 10, (0, 140, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, str(index), tuple(point + (12, -12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render een compacte visuele review van een veldprojectieplan.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True, help="Oorspronkelijk plan, inclusief verworpen matrices.")
    parser.add_argument("--player-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Toon uitsluitend tijdgespreide geldige projecties binnen het gekozen venster.",
    )
    args = parser.parse_args()

    video = args.video if args.video.is_absolute() else PROJECT_ROOT / args.video
    plan_path = args.plan if args.plan.is_absolute() else PROJECT_ROOT / args.plan
    report_path = args.player_report if args.player_report.is_absolute() else PROJECT_ROOT / args.player_report
    plan = load_video_projection_plan(plan_path)
    by_frame = {item.frame_number: item for item in plan.records}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = [
        record for record in report["records"]
        if (args.start is None or record["time_seconds"] >= args.start)
        and (args.end is None or record["time_seconds"] <= args.end)
    ]
    categories = (
        (("VALID | TIJDSPREIDING" if args.valid_only else "VALID + SPELERS AKKOORD"), [r for r in records if r["plan_status"] == "valid" and (args.valid_only or r["classification"] == "supportive")]),
    ) if args.valid_only else (
        ("VALID + SPELERS AKKOORD", [r for r in records if r["plan_status"] == "valid" and r["classification"] == "supportive"]),
        ("KANDIDAAT + TWIJFEL", [r for r in records if r["plan_status"] == "candidate" and r["classification"] == "ambiguous"]),
        ("DOOR SPELERS AFGEKEURD", [r for r in records if r["classification"] == "rejected"]),
    )
    selected = [(label, record) for label, records in categories for record in _evenly_spaced(records, args.per_category)]
    if not selected:
        raise RuntimeError("Geen geschikte reviewframes gevonden.")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Video kan niet worden geopend: {video}")
    profile = create_detection_profile(plan.match_format)
    tile_width, tile_height = 720, 405
    columns = args.per_category
    rows = int(np.ceil(len(selected) / columns))
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 24, dtype=np.uint8)
    colors = {"VALID | TIJDSPREIDING": (20, 190, 20), "VALID + SPELERS AKKOORD": (20, 190, 20), "KANDIDAAT + TWIJFEL": (0, 180, 255), "DOOR SPELERS AFGEKEURD": (20, 20, 230)}
    try:
        for index, (label, evidence) in enumerate(selected):
            planned = by_frame[int(evidence["frame_number"])]
            capture.set(cv2.CAP_PROP_POS_FRAMES, planned.frame_number)
            success, frame = capture.read()
            if not success or planned.projection is None:
                continue
            _draw_field(frame, planned.projection, profile.pitch_length_m, profile.pitch_width_m)
            frame = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            cv2.rectangle(frame, (0, 0), (tile_width, 58), (18, 18, 18), -1)
            cv2.putText(frame, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[label], 2, cv2.LINE_AA)
            detail = f'{evidence["time_seconds"]:.1f}s | voeten binnen marge {evidence["acceptable_ratio"]:.0%} | anker {planned.anchor_id}'
            cv2.putText(frame, detail, (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
            row, column = divmod(index, columns)
            sheet[row * tile_height:(row + 1) * tile_height, column * tile_width:(column + 1) * tile_width] = frame
    finally:
        capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"Preview kon niet worden opgeslagen: {args.output}")
    print(f"Reviewpreview: {args.output} | {len(selected)} frames")


if __name__ == "__main__":
    main()
