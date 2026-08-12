import json
from pathlib import Path
from tempfile import TemporaryDirectory

from football_ai.classification.goalkeeper_window_review import resolve_goalkeeper_window_reviews


def test_only_resolves_exact_current_human_reviewed_candidates():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        candidates = root / "candidates.json"
        reviews = root / "reviews.json"
        candidates.write_text(json.dumps({"windows": [
            {"goal": "A", "start_seconds": 1, "end_seconds": 2, "quality": {"classification": "consistent_review_candidate"}},
            {"goal": "B", "start_seconds": 3, "end_seconds": 4, "quality": {"classification": "ambiguous"}},
        ]}), encoding="utf-8")
        reviews.write_text(json.dumps({"human_reviewed": True, "reviews": [
            {"goal": "A", "start_seconds": 1, "end_seconds": 2, "answer": "keeper"},
            {"goal": "B", "start_seconds": 3, "end_seconds": 4, "answer": "not_keeper"},
            {"goal": "A", "start_seconds": 9, "end_seconds": 10, "answer": "keeper"},
        ]}), encoding="utf-8")
        result = resolve_goalkeeper_window_reviews(candidates, reviews)
    assert [(item.goal, item.answer) for item in result] == [("A", "keeper")]
