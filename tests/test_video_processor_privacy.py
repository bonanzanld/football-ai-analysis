from __future__ import annotations

import numpy as np

from football_ai.tracking.entity_corrections import EntityRole, TeamAssignment
from football_ai.tracking.entity_identity import PhysicalIdentity
from football_ai.tracking.entity_roster import PlayerProfile, TeamRoster
from football_ai.video_processor import VideoProcessor


class _Detector:
    pass


def test_video_processor_anonymizes_people_by_default():
    processor = VideoProcessor(detector=_Detector(), debug_homography=False)

    assert processor.anonymize_people is True


def test_video_processor_allows_explicit_internal_opt_out():
    processor = VideoProcessor(
        detector=_Detector(),
        debug_homography=False,
        anonymize_people=False,
    )

    assert processor.anonymize_people is False


def _identity_and_roster():
    identity = PhysicalIdentity(
        identity_id=3,
        label="Team B - Speler 2",
        track_ids=(7,),
        role=EntityRole.PLAYER,
        team=TeamAssignment.TEAM_B,
        first_frame=0,
        last_frame=100,
    )
    roster = TeamRoster(
        source_video="match.mp4",
        own_team_name="Brabantia",
        own_team=TeamAssignment.TEAM_B,
        players=(PlayerProfile(3, "Daan", "7"),),
    )
    return identity, roster


def test_anonymized_render_does_not_show_roster_name():
    identity, roster = _identity_and_roster()
    processor = VideoProcessor(detector=_Detector(), team_roster=roster)

    assert processor._identity_display_label(identity) == "Team B - Speler 2"


def test_internal_opt_out_may_show_roster_name():
    identity, roster = _identity_and_roster()
    processor = VideoProcessor(
        detector=_Detector(),
        team_roster=roster,
        anonymize_people=False,
    )

    assert processor._identity_display_label(identity) == "Brabantia - Daan (#7)"
