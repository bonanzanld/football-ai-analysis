from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2

from football_ai.calibration.bootstrap.goal_seed import (
    GoalSeedApp,
    create_goal_seed_preview,
    load_goal_seeds,
    save_goal_seeds,
)
from football_ai.calibration.bootstrap.detection_profile import create_detection_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Bevestig doelen en bouw de veldcontour.")
    parser.add_argument(
        "--contour-only",
        action="store_true",
        help="Gebruik bestaande doel-seeds en voeg alleen de veldcontour toe.",
    )
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--video", default="brandevoortbrab.mov", help="Bestandsnaam in de videos-map.")
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    if not video.exists():
        raise FileNotFoundError(f"Video niet gevonden: {video}")
    stem = video.stem
    bootstrap = PROJECT_ROOT / "output" / "pitch_bootstrap" / f"{stem}_bootstrap.json"
    profile = create_detection_profile(args.format)
    output = PROJECT_ROOT / "output" / "pitch_bootstrap" / f"{stem}_{args.format}_goal_seeds.json"
    preview_path = PROJECT_ROOT / "output" / "pitch_bootstrap" / f"{stem}_{args.format}_goal_seeds.jpg"
    seeds = (
        load_goal_seeds(output)
        if args.contour_only
        else GoalSeedApp(
            video,
            bootstrap,
            goal_width_m=profile.goal_width_m,
            match_format=profile.match_format.value,
            fallback_goal_times=_manual_goal_times(
                PROJECT_ROOT / "output" / "pitch_bootstrap" / f"{stem}_{args.format}_manual_perspective_reference.json"
            ),
        ).run()
    )
    save_goal_seeds(seeds, output, match_format=profile.match_format.value)
    preview = create_goal_seed_preview(video, seeds, pitch_width_m=profile.pitch_width_m)
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"QA-preview kon niet worden opgeslagen: {preview_path}")
    print(f"Doel-seeds opgeslagen: {output}")
    for seed in seeds:
        print(
            f"Doel {seed.goal_id}: frame {seed.frame_number} ({seed.time_seconds:.1f}s) | "
            f"achterlijnsteun {seed.backline_support:.1%}"
        )
    print(f"QA-preview: {preview_path}")


def _manual_goal_times(path: Path) -> tuple[float, float] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    times = {str(item["label"]): float(item["time_seconds"]) for item in data["views"]}
    if "left_goal" not in times or "right_goal" not in times:
        return None
    return times["left_goal"], times["right_goal"]


if __name__ == "__main__":
    main()
