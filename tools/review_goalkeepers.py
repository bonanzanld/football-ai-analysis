from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.tracking.entity_corrections import load_entity_corrections
from football_ai.tracking.entity_identity import load_entity_identities
from football_ai.tracking.entity_review_app import EntityReviewApp
from football_ai.tracking.entity_review_manifest import (
    EntityReviewManifest,
    load_entity_review_manifest,
)


def goalkeeper_review_manifest(
    manifest: EntityReviewManifest,
    candidate_track_ids: set[int],
) -> EntityReviewManifest:
    return EntityReviewManifest(
        source_video=manifest.source_video,
        fps=manifest.fps,
        tracks=tuple(
            track for track in manifest.tracks if track.track_id in candidate_track_ids
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controleer uitsluitend de korte lijst met mogelijke keepers.",
    )
    parser.add_argument("--video", required=True, help="Bestandsnaam in videos/ of volledig pad.")
    parser.add_argument("--team-a-name", default="Brandevoort groen-wit")
    parser.add_argument("--team-b-name", default="Brabantia rood-blauw")
    parser.add_argument(
        "--track-ids",
        help=(
            "Optioneel: controleer uitsluitend deze komma-gescheiden track-ID's, "
            "bijvoorbeeld --track-ids 6."
        ),
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / "videos" / video_path
    prefix = video_path.stem
    output_dir = PROJECT_ROOT / "output" / "entities"
    manifest_path = output_dir / f"{prefix}_entity_review.json"
    candidates_path = output_dir / f"{prefix}_goalkeeper_candidates.json"
    corrections_path = output_dir / f"{prefix}_entity_corrections.json"
    identities_path = output_dir / f"{prefix}_entity_identities.json"

    if not candidates_path.exists():
        raise FileNotFoundError(
            "Keeperrapport ontbreekt. Draai eerst tools/analyze_goalkeepers.py."
        )
    report = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidate_ids = (
        {
            int(value.strip())
            for value in args.track_ids.split(",")
            if value.strip()
        }
        if args.track_ids
        else {
            int(item["track_id"])
            for item in report.get("assessments", [])
            if item.get("decision") != "player"
        }
    )
    if not candidate_ids:
        raise ValueError("Het keeperrapport bevat geen personen die controle nodig hebben.")

    full_manifest = load_entity_review_manifest(manifest_path)
    review_manifest = goalkeeper_review_manifest(full_manifest, candidate_ids)
    existing = (
        load_entity_corrections(corrections_path) if corrections_path.exists() else None
    )
    app = EntityReviewApp(
        manifest=review_manifest,
        video_path=video_path,
        output_path=corrections_path,
        corrections=existing,
        minimum_frames_seen=1,
        team_a_name=args.team_a_name,
        team_b_name=args.team_b_name,
        identities=(
            load_entity_identities(identities_path) if identities_path.exists() else None
        ),
    )
    result = app.run()
    print(f"Keepercontrole opgeslagen: {corrections_path}")
    print(f"Totaal aanwezige handmatige beslissingen: {len(result.corrections)}")


if __name__ == "__main__":
    main()
