from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai.calibration.projection_boundary_review import (
    load_projection_boundary_reviews,
)
from football_ai.calibration.video_projection_plan import (
    gate_projection_plan_with_boundary_reviews,
    load_video_projection_plan,
    save_video_projection_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pas menselijke grensreviews conservatief toe op een veldprojectieplan."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = load_video_projection_plan(args.plan)
    reviews = load_projection_boundary_reviews(args.reviews)
    gated = gate_projection_plan_with_boundary_reviews(plan, reviews)
    save_video_projection_plan(gated, args.output)
    rejected = sum(
        before.projection_matrix is not None and after.projection_matrix is None
        for before, after in zip(plan.records, gated.records)
    )
    print(f"Menselijk afgekeurde volledige veldvlakken: {rejected}")
    print(f"Reviewplan: {args.output}")


if __name__ == "__main__":
    main()
