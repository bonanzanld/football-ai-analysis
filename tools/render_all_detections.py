from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detection.ball_tracking import BallObservation
from football_ai.visualizer import draw_ball_observation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combineer de actuele speler-, keeper-, voetpunt- en baldetecties.",
    )
    parser.add_argument("--video", required=True, help="Bestandsnaam in videos/ of volledig pad.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / "videos" / video_path
    prefix = video_path.stem
    entity_path = PROJECT_ROOT / "output" / "entities" / f"{prefix}_entities_qa.mp4"
    ball_path = PROJECT_ROOT / "output" / "ball" / f"{prefix}_ball_tracking.json"
    if not entity_path.exists():
        raise FileNotFoundError(
            f"Speler-/keepervideo ontbreekt: {entity_path}. Draai eerst tools/analyze_entities.py."
        )
    if not ball_path.exists():
        raise FileNotFoundError(
            f"Balrapport ontbreekt: {ball_path}. Draai eerst tools/analyze_ball.py."
        )

    observations = _load_ball_observations(ball_path)
    output_dir = PROJECT_ROOT / "output" / "combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{prefix}_all_detections_raw.mp4"
    output_path = output_dir / f"{prefix}_all_detections_qa.mp4"

    capture = cv2.VideoCapture(str(entity_path))
    if not capture.isOpened():
        raise RuntimeError(f"QA-video kon niet worden geopend: {entity_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    frame_number = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            annotated = draw_ball_observation(frame, observations.get(frame_number))
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(raw_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("Gecombineerde QA-video kon niet worden aangemaakt.")
            writer.write(annotated)
            frame_number += 1
            if frame_number % 30 == 0:
                print(f"Combineren {frame_number}/{total} frames")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if writer is None:
        raise RuntimeError("Geen frames gevonden in de entity-QA-video.")
    _transcode(raw_path, output_path)
    print(f"Frames gecombineerd: {frame_number}")
    print(f"Alle detecties: {output_path}")


def _load_ball_observations(path: Path) -> dict[int, BallObservation]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for item in payload.get("observations", []):
        observation = BallObservation(
            frame_number=int(item["frame_number"]),
            center=tuple(float(value) for value in item["center"]),
            box=tuple(float(value) for value in item["box"]),
            confidence=float(item["confidence"]),
            source=str(item["source"]),
        )
        result[observation.frame_number] = observation
    return result


def _transcode(raw_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
        raw_path.unlink()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError("Gecombineerde QA-video kon niet naar H.264 worden omgezet.") from error


if __name__ == "__main__":
    main()
