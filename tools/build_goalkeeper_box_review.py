from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.detection.goalkeeper_box_review import select_box_review_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Bouw een kleine boxgewijze keeperreview.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", default="8v8")
    parser.add_argument("--maximum-windows", type=int, default=4)
    args = parser.parse_args()
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{Path(args.video).stem}_{args.format}"
    source = json.loads((output / f"{prefix}_goalkeeper_review_candidates.json").read_text())
    result = select_box_review_candidates(source, args.maximum_windows)
    target = output / f"{prefix}_goalkeeper_box_review_candidates.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Boxreview: {len(result['examples'])} voorbeelden | {target}")


if __name__ == "__main__":
    main()
