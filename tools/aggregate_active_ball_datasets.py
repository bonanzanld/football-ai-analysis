from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detection.active_ball_dataset import (
    aggregate_active_ball_dataset_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate reviewed active-ball datasets across source clips."
    )
    parser.add_argument("--datasets", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-positive-frames", type=int, default=100)
    parser.add_argument("--minimum-source-clips", type=int, default=3)
    args = parser.parse_args()

    reports = []
    for path in args.datasets:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported dataset report schema: {path}")
        reports.append(payload)
    aggregate = aggregate_active_ball_dataset_reports(
        reports,
        minimum_positive_frames=args.minimum_positive_frames,
        minimum_source_clips=args.minimum_source_clips,
    )
    payload = {
        "schema_version": 1,
        "dataset_reports": [str(path) for path in args.datasets],
        **aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"source_clips={payload['source_clips']} | "
        f"positive_source_clips={payload['positive_source_clips']} | "
        f"positive_frames={payload['positive_frames']} | "
        f"ready_for_training={payload['ready_for_training']}"
    )
    for reason in payload["blocking_reasons"]:
        print(f"BLOCKED: {reason}")
    print(f"Aggregate dataset report: {args.output}")


if __name__ == "__main__":
    main()
