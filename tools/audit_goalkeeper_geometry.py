from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.classification.goalkeeper_geometry_audit import audit_goalkeeper_geometry


def main() -> None:
    root = PROJECT_ROOT / "data" / "goalkeeper_ground_truth"
    paths = tuple(sorted(root.glob("*_box_examples.json")))
    result = audit_goalkeeper_geometry(tuple(json.loads(path.read_text()) for path in paths))
    target = root / "geometry_audit.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    combined = result["combined"]
    print(
        f"Binnen doelmond: keeper {combined['keeper_inside_goal_mouth']} | "
        f"geen keeper {combined['not_keeper_inside_goal_mouth']}"
    )
    print(f"Geometrie-audit: {target}")


if __name__ == "__main__":
    main()
