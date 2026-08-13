from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from football_ai.analysis.entity_timeline import EntityTimeline, TimelineEntity
from football_ai.tracking.entity_corrections import EntityRole, TeamAssignment
from tools.analyze_possession import _anonymize_frame, _pseudonymize_timeline


@dataclass(frozen=True)
class _Entity:
    box: tuple[float, float, float, float]


def test_possession_anonymization_uses_timeline_entity_boxes():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[10:70, 30:60] = np.arange(60, dtype=np.uint8)[:, None, None]

    result = _anonymize_frame(frame, [_Entity((30, 10, 60, 70))])

    assert not np.array_equal(result[10:28, 30:60], frame[10:28, 30:60])
    np.testing.assert_array_equal(result[40:70, 30:60], frame[40:70, 30:60])


def test_possession_anonymization_handles_empty_frame():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    np.testing.assert_array_equal(_anonymize_frame(frame, []), frame)


def test_public_timeline_replaces_legacy_real_name():
    timeline = EntityTimeline(
        "match.mp4",
        30.0,
        (
            TimelineEntity(
                frame_number=4,
                track_id=17,
                identity_id=3,
                label="Brabantia - Daan (#7)",
                role=EntityRole.PLAYER,
                team=TeamAssignment.TEAM_B,
                box=(1.0, 2.0, 3.0, 4.0),
                footpoint=(2.0, 4.0),
            ),
        ),
    )

    public = _pseudonymize_timeline(timeline)

    assert public.observations[0].label == "team_b - Speler 3"
    assert "Daan" not in public.observations[0].label
