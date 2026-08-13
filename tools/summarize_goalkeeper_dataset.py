from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.detection.goalkeeper_dataset import summarize_goalkeeper_manifests


def main() -> None:
    root = PROJECT_ROOT / "data" / "goalkeeper_ground_truth"
    manifests = tuple(sorted(root.glob("*_window_examples.json")))
    result = summarize_goalkeeper_manifests(manifests)
    target = root / "summary.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Keeperdataset: {result['positive_examples']} positief | "
        f"{result['negative_examples']} negatief | {result['source_video_count']} bronvideo's"
    )
    print(f"Samenvatting: {target}")


if __name__ == "__main__":
    main()
