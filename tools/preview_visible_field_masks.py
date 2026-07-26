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
from football_ai.calibration.bootstrap.visible_field_mask import build_visible_field_mask


def main() -> None:
    video = PROJECT_ROOT / "videos" / "brandevoortbrab.mov"
    seeds = load_goal_seeds(PROJECT_ROOT / "output" / "pitch_bootstrap" / "brandevoortbrab_8v8_goal_seeds.json")
    capture = cv2.VideoCapture(str(video))
    tiles = []
    for seed in seeds:
        capture.set(cv2.CAP_PROP_POS_FRAMES, seed.frame_number)
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Frame {seed.frame_number} kon niet worden gelezen.")
        mask = build_visible_field_mask(seed, 42.5, (frame.shape[1], frame.shape[0]))
        points = np.round(mask.polygon).astype(np.int32)
        overlay = frame.copy(); cv2.fillPoly(overlay, [points], (30, 190, 30))
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0.0, frame)
        cv2.polylines(frame, [points], True, (0, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, f"DOEL {seed.goal_id} | zichtbaar veld {mask.frame_area_ratio:.0%}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(frame, (640, 360)))
    capture.release()
    output = PROJECT_ROOT / "output" / "pitch_bootstrap" / "brandevoortbrab_8v8_visible_field_masks.jpg"
    if not cv2.imwrite(str(output), np.hstack(tiles)):
        raise RuntimeError(f"Preview kon niet worden opgeslagen: {output}")
    print(f"Veldmasker-preview: {output}")


if __name__ == "__main__":
    main()
