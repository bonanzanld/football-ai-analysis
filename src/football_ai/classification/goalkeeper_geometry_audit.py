from __future__ import annotations


def audit_goalkeeper_geometry(manifests: tuple[dict, ...]) -> dict:
    sources = []
    combined = {"keeper": [0, 0], "not_keeper": [0, 0]}
    for payload in manifests:
        counts = {"keeper": [0, 0], "not_keeper": [0, 0]}
        for item in payload.get("examples", ()):
            label = "keeper" if item.get("label") in {"keeper", "goalkeeper"} else str(item.get("label"))
            position = item.get("goal_relative_position")
            if label not in counts or position is None:
                continue
            counts[label][1] += 1
            combined[label][1] += 1
            if _inside_goal_mouth(position):
                counts[label][0] += 1
                combined[label][0] += 1
        sources.append({
            "video_name": str(payload["video_name"]),
            "keeper_inside_goal_mouth": _fraction(counts["keeper"]),
            "not_keeper_inside_goal_mouth": _fraction(counts["not_keeper"]),
            "counts": _counts(counts),
        })
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "rule": "0 <= position_along_posts <= 1 and abs(distance_from_goal_line) <= 0.25 goal widths",
        "combined": {
            "keeper_inside_goal_mouth": _fraction(combined["keeper"]),
            "not_keeper_inside_goal_mouth": _fraction(combined["not_keeper"]),
            "counts": _counts(combined),
        },
        "sources": sources,
        "conclusion": "goal-relative position is a path-selection prior, not a goalkeeper classifier",
    }


def _inside_goal_mouth(position) -> bool:
    along, perpendicular = map(float, position)
    return 0.0 <= along <= 1.0 and abs(perpendicular) <= 0.25


def _fraction(values: list[int]) -> float | None:
    return values[0] / values[1] if values[1] else None


def _counts(groups: dict[str, list[int]]) -> dict:
    return {
        label: {"inside": values[0], "total": values[1]}
        for label, values in groups.items()
    }
