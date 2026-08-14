from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


_DECISIONS = {"confirmed", "rejected", "unknown"}


@dataclass(frozen=True, slots=True)
class ProjectionBoundaryReview:
    start_seconds: float
    end_seconds: float
    anchor_ids: tuple[str, ...]
    goal: str
    end_line: str
    sideline_front: str
    sideline_rear: str
    full_projection: str
    provenance: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.end_seconds < self.start_seconds:
            raise ValueError("Review end must not precede its start")
        decisions = (
            self.goal,
            self.end_line,
            self.sideline_front,
            self.sideline_rear,
            self.full_projection,
        )
        if any(value not in _DECISIONS for value in decisions):
            raise ValueError("Unknown boundary review decision")
        if self.provenance != "human_reviewed":
            raise ValueError("Boundary rejection requires human-reviewed provenance")

    def applies(self, time_seconds: float, anchor_id: str | None) -> bool:
        return (
            self.start_seconds <= time_seconds <= self.end_seconds
            and (not self.anchor_ids or anchor_id in self.anchor_ids)
        )


def load_projection_boundary_reviews(path: Path) -> tuple[ProjectionBoundaryReview, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError("Unsupported projection boundary review schema")
    return tuple(
        ProjectionBoundaryReview(
            start_seconds=float(item["start_seconds"]),
            end_seconds=float(item["end_seconds"]),
            anchor_ids=tuple(str(value) for value in item.get("anchor_ids", ())),
            goal=str(item.get("goal", "unknown")),
            end_line=str(item.get("end_line", "unknown")),
            sideline_front=str(item.get("sideline_front", "unknown")),
            sideline_rear=str(item.get("sideline_rear", "unknown")),
            full_projection=str(item.get("full_projection", "unknown")),
            provenance=str(item.get("provenance", "")),
            note=str(item.get("note", "")),
        )
        for item in payload.get("reviews", ())
    )
