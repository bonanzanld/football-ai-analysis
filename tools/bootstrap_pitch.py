from __future__ import annotations

import sys
from pathlib import Path
import argparse

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.calibration.bootstrap import PitchBootstrapAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="Ontdek representatieve camerastanden.")
    parser.add_argument("--video", default="brandevoortbrab.mov", help="Bestandsnaam in de videos-map.")
    args = parser.parse_args()
    video_path = PROJECT_ROOT / "videos" / args.video
    if not video_path.exists():
        raise FileNotFoundError(f"Video niet gevonden: {video_path}")
    stem = video_path.stem
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    report_path = output_dir / f"{stem}_bootstrap.json"
    preview_path = output_dir / f"{stem}_camera_states.jpg"

    print("=" * 66)
    print("Football AI - Pitch Bootstrap Sprint 4.1")
    print("=" * 66)
    analyzer = PitchBootstrapAnalyzer(
        duration_seconds=60.0,
        sample_interval_seconds=0.5,
        camera_state_count=7,
        coverage_scan_interval_seconds=5.0,
    )
    report = analyzer.analyze(video_path)
    preview = analyzer.create_contact_sheet(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    report.save_json(report_path)
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"Contactblad kon niet worden opgeslagen: {preview_path}")

    print(f"Geanalyseerde duur : {report.analyzed_duration_seconds:.1f}s")
    print(f"Dichte samples      : {report.dense_sample_count}")
    print(f"Samples incl. scan  : {len(report.samples)}")
    print(f"Camerastanden       : {len(report.clustering.clusters)}")
    print(
        "Stabiele standen     : "
        f"{sum(cluster.stable for cluster in report.clustering.clusters)}"
    )
    print(f"Clusterscheiding    : {report.clustering.separation_score:.1%}")
    for cluster in report.clustering.clusters:
        sample = report.samples[cluster.representative_sample_index]
        print(
            f"Stand {cluster.cluster_id} "
            f"({'STABIEL' if cluster.stable else 'ZELDZAAM'}): "
            f"{len(cluster.sample_indices)} samples | "
            f"representatief frame {sample.frame_number} ({sample.time_seconds:.1f}s)"
        )
    print(f"Rapport             : {report_path}")
    print(f"Contactblad         : {preview_path}")


if __name__ == "__main__":
    main()
