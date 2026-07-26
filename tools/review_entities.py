from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.tracking.entity_corrections import load_entity_corrections
from football_ai.tracking.entity_review_app import EntityReviewApp
from football_ai.tracking.entity_review_manifest import load_entity_review_manifest
from football_ai.tracking.entity_identity import load_entity_identities


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controleer spelers, keepers, scheidsrechter en uitgesloten personen.",
    )
    parser.add_argument("--video", required=True, help="Bestandsnaam in videos/ of volledig pad.")
    parser.add_argument(
        "--minimum-frames",
        type=int,
        default=30,
        help="Toon alleen tracks die minimaal zoveel frames zichtbaar waren.",
    )
    parser.add_argument(
        "--team-a-name",
        default="Team A (BLAUW kader)",
        help="Naam en herkenning van het team dat automatisch Team A heet.",
    )
    parser.add_argument(
        "--team-b-name",
        default="Team B (ROOD kader)",
        help="Naam en herkenning van het team dat automatisch Team B heet.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / "videos" / video_path
    prefix = video_path.stem
    output_dir = PROJECT_ROOT / "output" / "entities"
    manifest_path = output_dir / f"{prefix}_entity_review.json"
    corrections_path = output_dir / f"{prefix}_entity_corrections.json"
    identities_path = output_dir / f"{prefix}_entity_identities.json"

    if not video_path.exists():
        raise FileNotFoundError(f"Video niet gevonden: {video_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Reviewbestand niet gevonden. Voer eerst analyze_entities.py uit: "
            f"{manifest_path}"
        )

    manifest = load_entity_review_manifest(manifest_path)
    corrections = (
        load_entity_corrections(corrections_path)
        if corrections_path.exists()
        else None
    )
    app = EntityReviewApp(
        manifest=manifest,
        video_path=video_path,
        output_path=corrections_path,
        corrections=corrections,
        minimum_frames_seen=args.minimum_frames,
        team_a_name=args.team_a_name,
        team_b_name=args.team_b_name,
        identities=load_entity_identities(identities_path) if identities_path.exists() else None,
    )
    result = app.run()

    print(f"Correcties opgeslagen: {corrections_path}")
    print(f"Gecontroleerde tracks: {len(result.corrections)}")


if __name__ == "__main__":
    main()
