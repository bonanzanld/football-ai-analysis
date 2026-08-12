from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True, slots=True)
class GoalWindowPerson:
    frame_number: int
    footpoint: tuple[float, float]
    goal_proximity_score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class GoalPersonPathQuality:
    sample_count: int
    mean_goal_proximity: float
    maximum_step_ratio: float
    mean_step_ratio: float
    classification: str


def appearance_stability_classification(
    distances_to_median: tuple[float, ...],
    *,
    maximum_median_distance: float = 0.35,
) -> str:
    """Use within-window appearance only as a veto, never as identity proof."""
    if maximum_median_distance <= 0:
        raise ValueError("Maximum median distance must be positive")
    if len(distances_to_median) < 3:
        return "insufficient_evidence"
    values = sorted(float(item) for item in distances_to_median)
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0
    return "stable" if median <= maximum_median_distance else "unstable"


def evaluate_goal_person_path(
    path: tuple[GoalWindowPerson, ...],
    *,
    frame_diagonal: float,
) -> GoalPersonPathQuality:
    if frame_diagonal <= 0:
        raise ValueError("Frame diagonal must be positive")
    if not path:
        return GoalPersonPathQuality(0, 0.0, 0.0, 0.0, "unavailable")
    steps = tuple(
        hypot(second.footpoint[0] - first.footpoint[0], second.footpoint[1] - first.footpoint[1])
        / frame_diagonal
        for first, second in zip(path, path[1:])
    )
    proximity = sum(item.goal_proximity_score for item in path) / len(path)
    maximum_step = max(steps, default=0.0)
    mean_step = sum(steps) / max(1, len(steps))
    if len(path) < 3:
        classification = "insufficient_evidence"
    elif proximity >= 0.78 and maximum_step <= 0.055:
        classification = "consistent_review_candidate"
    else:
        classification = "ambiguous"
    return GoalPersonPathQuality(len(path), proximity, maximum_step, mean_step, classification)


def select_continuous_goal_person(
    frames: tuple[tuple[GoalWindowPerson, ...], ...],
    *,
    frame_diagonal: float,
    maximum_step_ratio: float = 0.08,
) -> tuple[GoalWindowPerson, ...]:
    """Select one smooth near-goal path without assigning the keeper role."""
    if frame_diagonal <= 0 or maximum_step_ratio <= 0:
        raise ValueError("Frame diagonal and maximum step ratio must be positive")
    if not frames or any(not frame for frame in frames):
        return ()
    costs = [-item.goal_proximity_score for item in frames[0]]
    parents: list[list[int]] = []
    for previous, current in zip(frames, frames[1:]):
        next_costs = []
        next_parents = []
        for candidate in current:
            options = []
            for index, other in enumerate(previous):
                distance_ratio = hypot(
                    candidate.footpoint[0] - other.footpoint[0],
                    candidate.footpoint[1] - other.footpoint[1],
                ) / frame_diagonal
                transition = distance_ratio / maximum_step_ratio
                if distance_ratio > maximum_step_ratio:
                    transition += 4.0
                options.append(costs[index] + transition - candidate.goal_proximity_score)
            parent = min(range(len(options)), key=options.__getitem__)
            next_parents.append(parent)
            next_costs.append(options[parent])
        parents.append(next_parents)
        costs = next_costs
    index = min(range(len(costs)), key=costs.__getitem__)
    path = [frames[-1][index]]
    for frame_index in range(len(frames) - 2, -1, -1):
        index = parents[frame_index][index]
        path.append(frames[frame_index][index])
    return tuple(reversed(path))
