from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.tracking.entity_corrections import (
    EntityCorrectionSet,
    load_entity_corrections,
    save_entity_corrections,
)
from football_ai.tracking.entity_review_app import (
    ACTIONS,
    EntityReviewApp,
    correction_for_action,
)
from football_ai.tracking.entity_review_manifest import (
    EntityReviewManifest,
    load_entity_review_manifest,
)
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
    parser.add_argument(
        "--segments",
        help=(
            "Optioneel: controleer uitsluitend deze komma-gescheiden segmenten, "
            "bijvoorbeeld --segments 55.2."
        ),
    )
    parser.add_argument(
        "--assign",
        choices=tuple(action.key for action in ACTIONS),
        help=(
            "Sla voor de opgegeven --segments direct een keuze op zonder de interface "
            "te openen, bijvoorbeeld --assign player_b."
        ),
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
    if args.segments:
        requested_segments = {
            tuple(int(part) for part in value.strip().split(".", maxsplit=1))
            for value in args.segments.split(",")
            if value.strip()
        }
        if any(len(value) != 2 for value in requested_segments):
            raise ValueError("Gebruik segmentnummers zoals 55.2, gescheiden door komma's.")
        manifest = EntityReviewManifest(
            source_video=manifest.source_video,
            fps=manifest.fps,
            tracks=tuple(
                track
                for track in manifest.tracks
                if (track.track_id, track.segment_index) in requested_segments
            ),
        )
        if not manifest.tracks:
            raise ValueError("Geen van de opgegeven segmenten staat in het reviewbestand.")
    corrections = (
        load_entity_corrections(corrections_path)
        if corrections_path.exists()
        else EntityCorrectionSet(source_video=manifest.source_video)
    )
    if args.assign:
        if not args.segments:
            raise ValueError("--assign vereist minimaal één waarde bij --segments.")
        for track in manifest.tracks:
            corrections = corrections.with_correction(
                correction_for_action(
                    track.track_id,
                    args.assign,
                    segment_index=track.segment_index,
                )
            )
        save_entity_corrections(corrections, corrections_path)
        assigned = ", ".join(
            f"{track.track_id}.{track.segment_index}" for track in manifest.tracks
        )
        print(f"Direct opgeslagen: {assigned} -> {args.assign}")
        print(f"Correcties opgeslagen: {corrections_path}")
        return
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
