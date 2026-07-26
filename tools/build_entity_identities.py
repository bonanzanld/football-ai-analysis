from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.tracking.entity_corrections import load_entity_corrections
from football_ai.tracking.entity_identity import build_entity_identities, save_entity_identities
from football_ai.tracking.entity_review_manifest import load_entity_review_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Koppel trackerfragmenten tot fysieke spelers.")
    parser.add_argument("--video", required=True, help="Bestandsnaam in videos/ of volledig pad.")
    parser.add_argument("--team-a-name", default="Team A")
    parser.add_argument("--team-b-name", default="Team B")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / "videos" / video_path
    output_dir = PROJECT_ROOT / "output" / "entities"
    prefix = video_path.stem
    manifest = load_entity_review_manifest(output_dir / f"{prefix}_entity_review.json")
    corrections_path = output_dir / f"{prefix}_entity_corrections.json"
    corrections = load_entity_corrections(corrections_path) if corrections_path.exists() else None
    result = build_entity_identities(
        manifest,
        video_path,
        corrections,
        team_a_name=args.team_a_name,
        team_b_name=args.team_b_name,
    )
    output_path = output_dir / f"{prefix}_entity_identities.json"
    save_entity_identities(result, output_path)
    merged = [item for item in result.identities if len(item.track_ids) > 1]
    candidates = sum(item.decision == "candidate" for item in result.links)
    print(f"Fysieke identiteiten: {len(result.identities)}")
    print(f"Automatisch samengevoegde personen: {len(merged)}")
    print(f"Twijfelgevallen (niet samengevoegd): {candidates}")
    print(f"Identiteitsbestand: {output_path}")


if __name__ == "__main__":
    main()
