import json

import pytest

from football_ai.calibration.projection_boundary_review import (
    ProjectionBoundaryReview,
    load_projection_boundary_reviews,
)


def test_partial_boundary_review_preserves_goal_and_rejects_field_plane(tmp_path):
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "reviews": [{
            "start_seconds": 902.0,
            "end_seconds": 916.5,
            "anchor_ids": ["local-30300"],
            "goal": "confirmed",
            "end_line": "confirmed",
            "sideline_front": "rejected",
            "sideline_rear": "rejected",
            "full_projection": "rejected",
            "provenance": "human_reviewed",
        }],
    }))

    review = load_projection_boundary_reviews(path)[0]

    assert review.applies(905.0, "local-30300")
    assert not review.applies(905.0, "goal-a")
    assert review.goal == "confirmed"
    assert review.full_projection == "rejected"


def test_boundary_rejection_cannot_masquerade_as_automatic_ground_truth():
    with pytest.raises(ValueError, match="human-reviewed"):
        ProjectionBoundaryReview(
            1.0, 2.0, (), "confirmed", "confirmed", "rejected", "rejected",
            "rejected", "automatic",
        )
