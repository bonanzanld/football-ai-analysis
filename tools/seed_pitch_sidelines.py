from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.calibration.bootstrap.goal_seed import load_goal_seeds
from football_ai.calibration.bootstrap.sideline_anchor import SidelineAnchorApp, save_sideline_anchors


def main() -> None:
    parser = argparse.ArgumentParser(description="Leg twee zijlijnpunten per tussenstand vast.")
    parser.add_argument("--video", default="brandevoortbrab.mov", help="Bestandsnaam in de videos-map.")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    if not video.exists():
        raise FileNotFoundError(f"Video niet gevonden: {video}")
    stem = video.stem
    bootstrap = PROJECT_ROOT / "output" / "pitch_bootstrap" / f"{stem}_bootstrap.json"
    goals_path = PROJECT_ROOT / "output" / "pitch_bootstrap" / f"{stem}_{args.format}_goal_seeds.json"
    output = PROJECT_ROOT / "output" / "pitch_bootstrap" / f"{stem}_{args.format}_sideline_anchors.json"
    goals = load_goal_seeds(goals_path)
    anchors = SidelineAnchorApp(
        video,
        bootstrap,
        excluded_states={seed.camera_state for seed in goals},
    ).run()
    save_sideline_anchors(anchors, output)
    print(f"Tussenankers opgeslagen: {output}")
    for anchor in anchors:
        visible = []
        if anchor.rear_point is not None:
            visible.append("verre zijlijn")
        if anchor.front_point is not None:
            visible.append("nabije zijlijn")
        print(
            f"Stand {anchor.camera_state}: frame {anchor.frame_number} ({anchor.time_seconds:.1f}s) | "
            f"waargenomen: {', '.join(visible) if visible else 'geen zijlijn'}"
        )


if __name__ == "__main__":
    main()
