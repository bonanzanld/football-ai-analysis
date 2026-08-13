from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.calibration.bootstrap.goal_seed import load_goal_seeds
from football_ai.calibration.bootstrap.local_mask_tracker import LocalMaskTracker
from football_ai.calibration.bootstrap.visible_field_mask import build_visible_field_mask
from football_ai.calibration.field_zone import FieldZone
from football_ai.detector import FootballDetector
from football_ai.privacy import anonymize_people_heads


COLORS = {FieldZone.INSIDE: (40, 210, 40), FieldZone.EDGE: (0, 165, 255), FieldZone.OUTSIDE: (40, 40, 230)}


def _anonymize_detected_people(frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    return anonymize_people_heads(frame, boxes)


def main() -> None:
    video_path = PROJECT_ROOT / "videos" / "brandevoortbrab.mov"
    seed_path = PROJECT_ROOT / "output" / "pitch_bootstrap" / "brandevoortbrab_8v8_goal_seeds.json"
    output_path = PROJECT_ROOT / "output" / "brandevoortbrab_visible_field_people_tracked_qa.mp4"
    seeds = load_goal_seeds(seed_path)
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"QA-video kon niet worden aangemaakt: {output_path}")
    detector = FootballDetector(player_threshold=0.20, ball_threshold=0.05)
    segment_frames = int(round(2.0 * fps))
    counts = {zone: 0 for zone in FieldZone}
    reliable_frames = 0
    fallback_frames = 0
    for seed in seeds:
        mask = build_visible_field_mask(seed, 42.5, (width, height))
        start = seed.frame_number
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        tracker = None
        for offset in range(segment_frames):
            success, frame = capture.read()
            if not success:
                break
            if tracker is None:
                tracker = LocalMaskTracker(frame, mask.polygon)
                polygon = mask.polygon
                tracking_label = "ankerframe"
            else:
                tracking = tracker.update(frame)
                polygon = tracking.polygon
                if tracking.reliable:
                    reliable_frames += 1
                    tracking_label = f"tracking OK {tracking.inlier_ratio:.0%}"
                else:
                    fallback_frames += 1
                    tracking_label = "tracking HOLD"
            _all, people, _balls = detector.detect(frame)
            frame = _anonymize_detected_people(frame, people.xyxy)
            points = np.round(polygon).astype(np.int32)
            cv2.polylines(frame, [points], True, (0, 255, 255), 3, cv2.LINE_AA)
            for box, confidence in zip(people.xyxy, people.confidence):
                x1, y1, x2, y2 = box.astype(float)
                foot = ((x1 + x2) / 2.0, y2)
                distance = cv2.pointPolygonTest(polygon.astype(np.float32), foot, True)
                zone = FieldZone.INSIDE if distance > 12.0 else (FieldZone.EDGE if distance >= -12.0 else FieldZone.OUTSIDE)
                counts[zone] += 1
                color = COLORS[zone]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(frame, (int(foot[0]), int(foot[1])), 5, color, -1, cv2.LINE_AA)
                cv2.putText(frame, f"{zone.value} {confidence:.2f}", (int(x1), max(20, int(y1) - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (0, 0), (width, 48), (20, 20, 20), -1)
            cv2.putText(frame, f"QA DOEL {seed.goal_id} | voetpunt bepaalt binnen/rand/buiten", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, tracking_label, (width - 270, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80,255,80) if "OK" in tracking_label or tracking_label == "ankerframe" else (0,165,255), 2, cv2.LINE_AA)
            writer.write(frame)
            if (offset + 1) % 30 == 0:
                print(f"Doel {seed.goal_id}: {offset + 1}/{segment_frames} frames")
    capture.release(); writer.release()
    print("Classificaties: " + " | ".join(f"{zone.value} {counts[zone]}" for zone in FieldZone))
    print(f"Mask-tracking: {reliable_frames} betrouwbaar | {fallback_frames} hold")
    print(f"QA-video: {output_path}")


if __name__ == "__main__":
    main()
