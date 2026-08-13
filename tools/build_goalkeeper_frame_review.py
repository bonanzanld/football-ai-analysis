from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.detection.goalkeeper_frame_review import select_frame_review_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", default="8v8")
    parser.add_argument("--maximum-per-goal", type=int, default=6)
    args = parser.parse_args()
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{Path(args.video).stem}_{args.format}"
    people = json.loads((output / f"{prefix}_goal_window_people_qa.json").read_text())
    review_path = output / f"{prefix}_goalkeeper_frame_reviews.json"
    required = frozenset()
    if review_path.exists():
        reviews = json.loads(review_path.read_text())
        required = frozenset(item["frame_id"] for item in reviews.get("reviews", ()))
    result = select_frame_review_candidates(people, args.maximum_per_goal, required)
    target = output / f"{prefix}_goalkeeper_frame_review_candidates.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Framereview: {len(result['frames'])} frames | {target}")


if __name__ == "__main__":
    main()
