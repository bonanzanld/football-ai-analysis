from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.classification.goalkeeper_analysis import (
    analyze_goalkeeper_candidates,
    save_goalkeeper_analysis,
    shortlist_goalkeeper_assessments,
)
from football_ai.tracking.entity_review_manifest import load_entity_review_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rangschik keeperkandidaten op tenue, trackhistorie en beweging.",
    )
    parser.add_argument("--video", required=True, help="Bestandsnaam in videos/ of volledig pad.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / "videos" / video_path
    if not video_path.exists():
        raise FileNotFoundError(f"Video niet gevonden: {video_path}")

    entity_dir = PROJECT_ROOT / "output" / "entities"
    manifest_path = entity_dir / f"{video_path.stem}_entity_review.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Entity-review ontbreekt: {manifest_path}. Draai eerst tools/analyze_entities.py."
        )
    report = analyze_goalkeeper_candidates(
        video_path,
        load_entity_review_manifest(manifest_path),
    )
    output_path = entity_dir / f"{video_path.stem}_goalkeeper_candidates.json"
    save_goalkeeper_analysis(report, output_path)

    candidates = shortlist_goalkeeper_assessments(report.assessments)
    print(f"Keeperkandidaten: {len(candidates)}")
    for item in candidates:
        team = "onbekend" if item.team_id is None else f"team {item.team_id + 1}"
        reasons = ", ".join(item.reasons) or "onvoldoende gecombineerd bewijs"
        print(
            f"Track {item.track_id} | {team} | {item.decision.value} | "
            f"score {item.score:.0%} | {reasons}"
        )
    print("Doelreferentie: niet nodig voor deze keeperselectie.")
    print(f"Keeperrapport: {output_path}")


if __name__ == "__main__":
    main()
