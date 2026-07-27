from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.classification.participant_analysis import (
    analyze_participants,
    review_candidates,
    save_participant_analysis,
)
from football_ai.tracking.entity_corrections import load_entity_corrections
from football_ai.tracking.entity_review_manifest import load_entity_review_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selecteer conservatief scheidsrechter- en buitenstaanderkandidaten.",
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
    correction_path = entity_dir / f"{video_path.stem}_entity_corrections.json"
    corrections = (
        load_entity_corrections(correction_path) if correction_path.exists() else None
    )
    report = analyze_participants(
        load_entity_review_manifest(manifest_path),
        corrections,
    )
    output_path = entity_dir / f"{video_path.stem}_participant_candidates.json"
    save_participant_analysis(report, output_path)

    candidates = review_candidates(report.assessments)
    print(f"Te controleren personen: {len(candidates)}")
    for item in candidates:
        segment = "" if item.segment_index is None else f".{item.segment_index}"
        reasons = ", ".join(item.reasons) or "onvoldoende gecombineerd bewijs"
        print(
            f"Track {item.track_id}{segment} | {item.decision.value} | "
            f"score {item.score:.0%} | {reasons}"
        )
    print("Er is niemand automatisch uitgesloten of als scheidsrechter opgeslagen.")
    print(f"Deelnemerrapport: {output_path}")


if __name__ == "__main__":
    main()
