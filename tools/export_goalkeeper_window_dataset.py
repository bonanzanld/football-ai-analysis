from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.detection.goalkeeper_dataset import build_goalkeeper_window_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporteer menselijk beoordeelde keepervensterboxen.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", default="8v8")
    args = parser.parse_args()
    prefix = f"{Path(args.video).stem}_{args.format}"
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    payload = build_goalkeeper_window_dataset(output / f"{prefix}_resolved_goalkeeper_windows.json")
    path = PROJECT_ROOT / "data" / "goalkeeper_ground_truth" / f"{prefix}_window_examples.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    positives = sum(item["label"] == "goalkeeper" for item in payload["examples"])
    print(f"Keepervoorbeelden: {positives} positief (alleen daadwerkelijk getoonde 3-uit-3-beelden) | 0 negatief")
    print(f"Datasetmanifest: {path}")


if __name__ == "__main__":
    main()
