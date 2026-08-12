from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ReviewedGoalkeeperWindow:
    goal: str
    start_seconds: float
    end_seconds: float
    answer: str
    candidate: dict


def resolve_goalkeeper_window_reviews(
    candidates_path: Path,
    reviews_path: Path,
) -> tuple[ReviewedGoalkeeperWindow, ...]:
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
    if not reviews.get("human_reviewed", False):
        raise ValueError("Keepervensterbeslissingen moeten expliciet menselijk beoordeeld zijn.")
    answers = {
        _key(item): str(item["answer"])
        for item in reviews.get("reviews", ())
    }
    allowed = {"keeper", "not_keeper", "uncertain"}
    resolved = []
    for candidate in candidates.get("windows", ()):
        if candidate.get("quality", {}).get("classification") != "consistent_review_candidate":
            continue
        key = _key(candidate)
        answer = answers.get(key)
        if answer is None:
            continue
        if answer not in allowed:
            raise ValueError(f"Onbekend keepervensterantwoord: {answer}")
        resolved.append(
            ReviewedGoalkeeperWindow(key[0], key[1], key[2], answer, candidate)
        )
    return tuple(resolved)


def _key(item: dict) -> tuple[str, float, float]:
    return (
        str(item["goal"]),
        round(float(item["start_seconds"]), 3),
        round(float(item["end_seconds"]), 3),
    )
