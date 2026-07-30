from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detector import FootballDetector
from football_ai.tracking.entity_corrections import load_entity_corrections
from football_ai.tracking.entity_identity import load_entity_identities
from football_ai.tracking.entity_roster import load_team_roster
from football_ai.video_processor import VideoProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detecteer en volg voetbalentiteiten zonder pitchkalibratie.",
    )
    parser.add_argument("--video", required=True, help="Bestandsnaam in videos/ of volledig pad.")
    parser.add_argument("--seconds", type=float, default=30.0, help="Aantal testseconden.")
    parser.add_argument("--corrections", type=Path, help="Optioneel JSON-bestand met reviewcorrecties.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / "videos" / video_path
    if not video_path.exists():
        raise FileNotFoundError(f"Video niet gevonden: {video_path}")

    output_dir = PROJECT_ROOT / "output" / "entities"
    output_path = output_dir / f"{video_path.stem}_entities_qa.mp4"
    manifest_path = output_dir / f"{video_path.stem}_entity_review.json"
    segmentation_path = output_dir / f"{video_path.stem}_track_segments.json"
    default_corrections_path = output_dir / f"{video_path.stem}_entity_corrections.json"
    identities_path = output_dir / f"{video_path.stem}_entity_identities.json"
    timeline_path = output_dir / f"{video_path.stem}_entity_timeline.json"
    roster_path = output_dir / f"{video_path.stem}_team_roster.json"
    corrections_path = args.corrections or (
        default_corrections_path if default_corrections_path.exists() else None
    )
    corrections = (
        load_entity_corrections(corrections_path)
        if corrections_path is not None
        else None
    )
    identities = load_entity_identities(identities_path) if identities_path.exists() else None
    roster = load_team_roster(roster_path) if roster_path.exists() else None

    if corrections is not None:
        print(f"Bestaande persoonscorrecties worden toegepast: {corrections_path}")
    if identities is not None:
        print(f"Fysieke spelersidentiteiten worden toegepast: {identities_path}")
    if roster is not None:
        print(f"Spelersnamen van het eigen team worden toegepast: {roster_path}")

    processor = VideoProcessor(
        detector=FootballDetector(player_threshold=0.20, ball_threshold=0.05),
        pitch_calibration=None,
        debug_homography=False,
        entity_corrections=corrections,
        entity_identities=identities,
        team_roster=roster,
    )
    frames = processor.process(
        video_path=video_path,
        output_path=output_path,
        max_seconds=args.seconds,
        review_manifest_path=manifest_path,
        segmentation_path=segmentation_path,
        entity_timeline_path=timeline_path,
        stable_team_render=True,
    )

    print(f"Frames verwerkt: {frames}")
    print(f"QA-video: {output_path}")
    print(f"Reviewbestand: {manifest_path}")
    print(f"Tracksegmenten: {segmentation_path}")
    print(f"Entiteitentijdlijn: {timeline_path}")


if __name__ == "__main__":
    main()
