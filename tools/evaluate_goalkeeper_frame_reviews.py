from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.detection.goalkeeper_frame_review import evaluate_frame_reviews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", default="8v8")
    args = parser.parse_args()
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{Path(args.video).stem}_{args.format}"
    load = lambda suffix: json.loads((output / f"{prefix}_{suffix}.json").read_text())
    result = evaluate_frame_reviews(
        load("goalkeeper_frame_review_candidates"), load("goalkeeper_frame_reviews"),
        load("goalkeeper_review_candidates"),
    )
    target = output / f"{prefix}_goalkeeper_frame_evaluation.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
