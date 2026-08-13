from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.detection.goalkeeper_box_review import build_box_review_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", default="8v8")
    args = parser.parse_args()
    prefix = f"{Path(args.video).stem}_{args.format}"
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    result = build_box_review_examples(
        json.loads((output / f"{prefix}_goalkeeper_box_review_candidates.json").read_text()),
        json.loads((output / f"{prefix}_goalkeeper_box_reviews.json").read_text()),
    )
    target = PROJECT_ROOT / "data" / "goalkeeper_ground_truth" / f"{prefix}_box_examples.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    positive = sum(item["label"] == "keeper" for item in result["examples"])
    negative = sum(item["label"] == "not_keeper" for item in result["examples"])
    print(f"Keeperboxlabels: {positive} positief | {negative} negatief | {target}")


if __name__ == "__main__":
    main()
