from __future__ import annotations

import supervision as sv


class FootballTracker:
    """Volgt gedetecteerde spelers met ByteTrack."""

    def __init__(
        self,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 60,
        minimum_matching_threshold: float = 0.80,
    ) -> None:
        self.tracker = sv.ByteTrack(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )

    def update(
        self,
        player_detections: sv.Detections,
    ) -> sv.Detections:
        """
        Geef detecties door aan ByteTrack.

        De teruggegeven detecties bevatten tracker_id's.
        """

        if len(player_detections) == 0:
            return player_detections

        return self.tracker.update_with_detections(
            player_detections
        )

    def reset(self) -> None:
        """Reset de tracker voor een nieuwe video."""
        self.tracker.reset()