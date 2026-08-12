from __future__ import annotations

from statistics import mean


FEATURES = (
    "mean_goal_proximity",
    "maximum_step_ratio",
    "mean_step_ratio",
    "appearance_median_distance",
)


def audit_reviewed_goalkeeper_windows(payload: dict) -> dict:
    """Describe reviewed selection windows without treating them as box negatives."""
    positive = tuple(payload.get("accepted_keeper_windows", ()))
    rejected = tuple(payload.get("rejected_keeper_windows", ()))
    uncertain = tuple(payload.get("uncertain_keeper_windows", ()))

    feature_summary = {}
    for feature in FEATURES:
        positive_values = _values(positive, feature)
        rejected_values = _values(rejected, feature)
        feature_summary[feature] = {
            "keeper_mean": mean(positive_values) if positive_values else None,
            "not_three_of_three_mean": mean(rejected_values) if rejected_values else None,
            "keeper_range": _range(positive_values),
            "not_three_of_three_range": _range(rejected_values),
            "ranges_overlap": _overlap(positive_values, rejected_values),
        }

    return {
        "schema_version": 1,
        "evaluation_unit": "review_window",
        "review_semantics": {
            "keeper": "the selected person is the correct goalkeeper in all three displayed frames",
            "not_keeper": "the three-of-three condition failed; individual negative boxes are unknown",
        },
        "counts": {
            "keeper": len(positive),
            "not_three_of_three": len(rejected),
            "uncertain": len(uncertain),
        },
        "feature_summary": feature_summary,
        "limitations": [
            "selection-conditioned: only automatically proposed windows were reviewed",
            "same-video diagnostic: this does not measure generalisation to another match",
            "not_keeper windows are not valid negative box labels",
            "sample size is too small for accuracy, precision, recall, or model training",
        ],
    }


def _values(windows: tuple[dict, ...], feature: str) -> tuple[float, ...]:
    return tuple(
        float(window["quality"][feature])
        for window in windows
        if window.get("quality", {}).get(feature) is not None
    )


def _range(values: tuple[float, ...]) -> list[float] | None:
    return [min(values), max(values)] if values else None


def _overlap(first: tuple[float, ...], second: tuple[float, ...]) -> bool | None:
    if not first or not second:
        return None
    return max(min(first), min(second)) <= min(max(first), max(second))
