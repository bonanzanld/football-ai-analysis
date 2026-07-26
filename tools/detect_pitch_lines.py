from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.calibration.bootstrap.detection_profile import create_detection_profile
from football_ai.calibration.bootstrap.line_detection_analyzer import BootstrapLineDetectionAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecteer witte lijnen per bootstrap-camerastand.")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    args = parser.parse_args()
    video_path = PROJECT_ROOT / "videos" / "brandevoortbrab.mov"
    bootstrap_path = PROJECT_ROOT / "output" / "pitch_bootstrap" / "brandevoortbrab_bootstrap.json"
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    json_path = output_dir / f"brandevoortbrab_{args.format}_white_lines.json"
    preview_path = output_dir / f"brandevoortbrab_{args.format}_white_lines.jpg"

    profile = create_detection_profile(args.format)
    analyzer = BootstrapLineDetectionAnalyzer(profile)
    results = analyzer.analyze(video_path, bootstrap_path)
    pair_selection = analyzer.select_goal_pair(results)
    analyzer.save_json(results, json_path)
    preview = analyzer.create_contact_sheet(results)
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"Preview kon niet worden opgeslagen: {preview_path}")

    print("=" * 66)
    print(f"Football AI - Witte-lijndetectie ({profile.name})")
    print("=" * 66)
    for item in results:
        print(
            f"Stand {item.camera_state}: {len(item.detection.candidates)} kandidaten | "
            f"doelen {len(item.goal_detection.candidates)} | "
            f"bevestigd {len(item.confirmed_goals)} | "
            f"gras {item.detection.grass_coverage:.1%} | "
            f"wit {item.detection.white_pixel_ratio:.2%}"
        )
    print(f"Doelpaar : {pair_selection.reason}")
    print(f"Rapport : {json_path}")
    print(f"Preview : {preview_path}")


if __name__ == "__main__":
    main()
