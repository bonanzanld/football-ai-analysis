from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detector import FootballDetector
from football_ai.filtering.player_filter import PlayerFilter
from football_ai.tracker import FootballTracker
from football_ai.tracking.track_engine import TrackEngine


CONFIGURATIONS = {
    "baseline": dict(track_activation_threshold=0.25, lost_track_buffer=60, minimum_matching_threshold=0.80),
    "longer_memory": dict(track_activation_threshold=0.25, lost_track_buffer=120, minimum_matching_threshold=0.80),
    "permissive_matching": dict(track_activation_threshold=0.25, lost_track_buffer=90, minimum_matching_threshold=0.90),
    "cleaner_activation": dict(track_activation_threshold=0.32, lost_track_buffer=90, minimum_matching_threshold=0.85),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Vergelijk trackerinstellingen met exact dezelfde persoonsdetecties.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--seconds", type=float, default=15.0)
    args = parser.parse_args()
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / "videos" / video_path

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Video kon niet worden geopend: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    limit = max(1, round(args.seconds * fps))
    detector = FootballDetector(player_threshold=0.20, ball_threshold=0.05)
    player_filter = PlayerFilter(
        minimum_box_height=24,
        minimum_aspect_ratio=1.15,
        maximum_aspect_ratio=6.0,
        minimum_foot_y_ratio=0.15,
        minimum_green_ratio=0.18,
        pitch_calibration=None,
    )
    trackers = {
        name: FootballTracker(frame_rate=fps, **configuration)
        for name, configuration in CONFIGURATIONS.items()
    }
    engines = {name: TrackEngine(field_projector=None) for name in trackers}
    frame_number = 0
    try:
        while frame_number < limit:
            success, frame = capture.read()
            if not success:
                break
            _, people, _ = detector.detect(frame)
            people = player_filter.filter(frame, people, frame_number)
            for name, tracker in trackers.items():
                tracked = tracker.update(people)
                engines[name].update(tracked, frame_number)
            frame_number += 1
            if frame_number % 150 == 0:
                print(f"Vergelijking: {frame_number}/{limit} frames")
    finally:
        capture.release()

    report = {"source_video": str(video_path), "frames": frame_number, "configurations": {}}
    print("\nTrackervergelijking")
    print("-" * 78)
    print(f"{'variant':24} {'tracks':>7} {'noise':>7} {'usable':>7} {'stable':>7} {'lang':>7}")
    for name, engine in engines.items():
        engine.finalize()
        evaluations = list(engine.track_evaluations.values())
        tracks = engine.tracks
        values = {
            "configuration": CONFIGURATIONS[name],
            "tracks": len(tracks),
            "noise": sum(item.is_noise for item in evaluations),
            "usable": sum(item.is_usable for item in evaluations),
            "stable": sum(item.is_stable for item in evaluations),
            "long_tracks": sum(item.frames_seen >= 90 for item in tracks),
        }
        report["configurations"][name] = values
        print(
            f"{name:24} {values['tracks']:7d} {values['noise']:7d} "
            f"{values['usable']:7d} {values['stable']:7d} {values['long_tracks']:7d}"
        )
    output = PROJECT_ROOT / "output" / "entities" / f"{video_path.stem}_tracker_benchmark.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport: {output}")


if __name__ == "__main__":
    main()
