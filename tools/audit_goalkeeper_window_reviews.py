from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.classification.goalkeeper_window_audit import audit_reviewed_goalkeeper_windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnoseer menselijk beoordeelde keepervensters.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", default="8v8")
    args = parser.parse_args()
    output = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{Path(args.video).stem}_{args.format}"
    source = output / f"{prefix}_resolved_goalkeeper_windows.json"
    result = audit_reviewed_goalkeeper_windows(json.loads(source.read_text(encoding="utf-8")))
    target = output / f"{prefix}_goalkeeper_window_audit.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    counts = result["counts"]
    print(
        f"Vensters: {counts['keeper']} keeper | "
        f"{counts['not_three_of_three']} niet 3-uit-3 | {counts['uncertain']} onzeker"
    )
    print(f"Diagnose: {target}")


if __name__ == "__main__":
    main()
