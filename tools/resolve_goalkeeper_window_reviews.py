from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.classification.goalkeeper_window_review import resolve_goalkeeper_window_reviews


def main() -> None:
    parser = argparse.ArgumentParser(description="Verwerk menselijk beoordeelde keepervensters.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", default="8v8")
    args = parser.parse_args()
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{Path(args.video).stem}_{args.format}"
    candidates = output / f"{prefix}_goalkeeper_review_candidates.json"
    reviews = output / f"{prefix}_goalkeeper_window_reviews.json"
    resolved = resolve_goalkeeper_window_reviews(candidates, reviews)
    payload = {
        "schema_version": 1,
        "video_name": Path(args.video).name,
        "human_reviewed": True,
        "accepted_keeper_windows": [item.candidate for item in resolved if item.answer == "keeper"],
        "rejected_keeper_windows": [item.candidate for item in resolved if item.answer == "not_keeper"],
        "uncertain_keeper_windows": [item.candidate for item in resolved if item.answer == "uncertain"],
    }
    path = output / f"{prefix}_resolved_goalkeeper_windows.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Bevestigd keeper: {len(payload['accepted_keeper_windows'])} | "
        f"geen keeper: {len(payload['rejected_keeper_windows'])} | "
        f"onzeker: {len(payload['uncertain_keeper_windows'])}"
    )
    print(f"Verwerkte keepervensters: {path}")


if __name__ == "__main__":
    main()
