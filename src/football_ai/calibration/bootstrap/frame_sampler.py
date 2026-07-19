from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class BootstrapFrameSample:
    sample_index: int
    frame_number: int
    time_seconds: float
    frame: np.ndarray


def sample_bootstrap_frames(
    video_path: Path,
    duration_seconds: float = 60.0,
    interval_seconds: float = 0.5,
    preview_width: int = 640,
) -> tuple[list[BootstrapFrameSample], float, int]:
    """Lees gelijkmatig verdeelde frames uit het begin van een video."""
    if duration_seconds <= 0.0 or interval_seconds <= 0.0:
        raise ValueError("Duur en sample-interval moeten groter zijn dan nul.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video kon niet worden geopend: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0.0 or total_frames <= 0:
        capture.release()
        raise RuntimeError("Video heeft ongeldige FPS- of framedata.")
    maximum_frame = min(total_frames - 1, int(round(duration_seconds * fps)))
    frame_step = max(1, int(round(interval_seconds * fps)))
    samples: list[BootstrapFrameSample] = []
    for frame_number in range(0, maximum_frame + 1, frame_step):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
        if not success:
            continue
        if preview_width > 0 and frame.shape[1] > preview_width:
            scale = preview_width / frame.shape[1]
            frame = cv2.resize(
                frame,
                (preview_width, int(round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        samples.append(
            BootstrapFrameSample(
                sample_index=len(samples),
                frame_number=frame_number,
                time_seconds=frame_number / fps,
                frame=frame,
            )
        )
    capture.release()
    if len(samples) < 3:
        raise RuntimeError("Bootstrap bevat minder dan drie leesbare samples.")
    return samples, fps, total_frames
