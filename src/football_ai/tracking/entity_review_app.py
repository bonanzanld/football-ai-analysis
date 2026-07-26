from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .entity_corrections import (
    EntityCorrectionSet,
    EntityRole,
    TeamAssignment,
    TrackCorrection,
    save_entity_corrections,
)
from .entity_review_manifest import EntityReviewManifest, ReviewTrack
from .entity_identity import EntityIdentitySet, grouped_review_tracks


@dataclass(frozen=True, slots=True)
class ReviewAction:
    key: str
    label: str
    color: tuple[int, int, int]


ACTIONS = (
    ReviewAction("player_a", "SPELER TEAM A", (255, 120, 0)),
    ReviewAction("goalkeeper_a", "KEEPER TEAM A", (255, 180, 0)),
    ReviewAction("player_b", "SPELER TEAM B", (0, 0, 255)),
    ReviewAction("goalkeeper_b", "KEEPER TEAM B", (0, 80, 255)),
    ReviewAction("referee", "SCHEIDSRECHTER", (255, 0, 255)),
    ReviewAction("exclude", "UITSLUITEN", (100, 100, 100)),
    ReviewAction("unknown", "ONBEKEND", (0, 220, 220)),
)


def correction_for_action(
    track_id: int,
    action: str,
    segment_index: int | None = None,
) -> TrackCorrection:
    values = {
        "player_a": (EntityRole.PLAYER, TeamAssignment.TEAM_A, False),
        "goalkeeper_a": (EntityRole.GOALKEEPER, TeamAssignment.TEAM_A, False),
        "player_b": (EntityRole.PLAYER, TeamAssignment.TEAM_B, False),
        "goalkeeper_b": (EntityRole.GOALKEEPER, TeamAssignment.TEAM_B, False),
        "referee": (EntityRole.REFEREE, TeamAssignment.OFFICIAL, False),
        "exclude": (EntityRole.STAFF, TeamAssignment.NONE, True),
        "unknown": (EntityRole.UNKNOWN, TeamAssignment.UNKNOWN, False),
    }
    if action not in values:
        raise ValueError(f"Onbekende reviewactie: {action}")
    role, team, excluded = values[action]
    return TrackCorrection(
        track_id=track_id,
        segment_index=segment_index,
        role=role,
        team=team,
        excluded=excluded,
    )


class EntityReviewApp:
    WINDOW_NAME = "Football AI - personen controleren"
    CANVAS_WIDTH = 1600
    CANVAS_HEIGHT = 900
    VIDEO_WIDTH = 1180
    SIDEBAR_X = 1180

    def __init__(
        self,
        manifest: EntityReviewManifest,
        video_path: Path,
        output_path: Path,
        corrections: EntityCorrectionSet | None = None,
        minimum_frames_seen: int = 30,
        team_a_name: str = "Team A",
        team_b_name: str = "Team B",
        identities: EntityIdentitySet | None = None,
    ) -> None:
        self.manifest = manifest
        self.video_path = video_path
        self.output_path = output_path
        segmented_tracks = tuple(
            track for track in manifest.tracks if track.segment_index is not None
        )
        if identities is not None and segmented_tracks:
            segmented_ids = {track.track_id for track in segmented_tracks}
            unsplit_manifest = EntityReviewManifest(
                source_video=manifest.source_video,
                fps=manifest.fps,
                tracks=tuple(
                    track
                    for track in manifest.tracks
                    if track.segment_index is None and track.track_id not in segmented_ids
                ),
            )
            source_tracks = (
                *grouped_review_tracks(unsplit_manifest, identities),
                *segmented_tracks,
            )
        elif identities is not None:
            source_tracks = grouped_review_tracks(manifest, identities)
        else:
            source_tracks = manifest.tracks
        self.identities = identities
        self.tracks = sorted(
            (
                track
                for track in source_tracks
                if track.frames_seen >= minimum_frames_seen and track.observations
            ),
            key=lambda track: (-track.frames_seen, track.track_id),
        )
        if not self.tracks:
            raise ValueError("Geen reviewbare tracks gevonden.")
        self.corrections = corrections or EntityCorrectionSet(
            source_video=manifest.source_video
        )
        if self.corrections.source_video != manifest.source_video:
            raise ValueError("Correctiebestand hoort bij een andere bronvideo.")
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Video kon niet worden geopend: {video_path}")
        self.track_index = 0
        self.sample_index = 0
        self.button_regions: list[tuple[tuple[int, int, int, int], str]] = []
        self.running = True
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name

    def run(self) -> EntityCorrectionSet:
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, self.CANVAS_WIDTH, self.CANVAS_HEIGHT)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)
        try:
            while self.running:
                cv2.imshow(self.WINDOW_NAME, self._render())
                key = cv2.waitKeyEx(30)
                self._handle_key(key)
        finally:
            self.capture.release()
            cv2.destroyWindow(self.WINDOW_NAME)
            self._save()
        return self.corrections

    @property
    def current_track(self) -> ReviewTrack:
        return self.tracks[self.track_index]

    def _render(self) -> np.ndarray:
        track = self.current_track
        observation = track.observations[self.sample_index]
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, observation.frame_number)
        success, frame = self.capture.read()
        if not success:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas = np.full(
            (self.CANVAS_HEIGHT, self.CANVAS_WIDTH, 3),
            (22, 22, 22),
            dtype=np.uint8,
        )
        displayed, scale, offset_x, offset_y = self._fit_frame(frame)
        canvas[offset_y:offset_y + displayed.shape[0], offset_x:offset_x + displayed.shape[1]] = displayed
        x1, y1, x2, y2 = observation.box
        start = (int(x1 * scale) + offset_x, int(y1 * scale) + offset_y)
        end = (int(x2 * scale) + offset_x, int(y2 * scale) + offset_y)
        cv2.rectangle(canvas, start, end, (0, 255, 255), 4)
        detail = self._create_detail(frame, observation.box)
        self._draw_sidebar(canvas, track, observation.frame_number, detail)
        return canvas

    @staticmethod
    def _create_detail(
        frame: np.ndarray,
        box: tuple[float, float, float, float],
    ) -> np.ndarray:
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = box
        width = max(x2 - x1, 20.0)
        height = max(y2 - y1, 40.0)
        margin_x = width * 1.5
        margin_y = height * 0.7
        crop_x1 = max(0, int(x1 - margin_x))
        crop_y1 = max(0, int(y1 - margin_y))
        crop_x2 = min(frame_width, int(x2 + margin_x))
        crop_y2 = min(frame_height, int(y2 + margin_y))
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if crop.size == 0:
            return np.zeros((150, 360, 3), dtype=np.uint8)
        scale = min(360 / crop.shape[1], 150 / crop.shape[0])
        resized = cv2.resize(
            crop,
            (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
        )
        detail = np.zeros((150, 360, 3), dtype=np.uint8)
        offset_x = (360 - resized.shape[1]) // 2
        offset_y = (150 - resized.shape[0]) // 2
        detail[offset_y:offset_y + resized.shape[0], offset_x:offset_x + resized.shape[1]] = resized
        return detail

    def _fit_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = frame.shape[:2]
        scale = min(self.VIDEO_WIDTH / width, self.CANVAS_HEIGHT / height)
        resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
        offset_x = (self.VIDEO_WIDTH - resized.shape[1]) // 2
        offset_y = (self.CANVAS_HEIGHT - resized.shape[0]) // 2
        return resized, scale, offset_x, offset_y

    def _draw_sidebar(
        self,
        canvas: np.ndarray,
        track: ReviewTrack,
        frame_number: int,
        detail: np.ndarray,
    ) -> None:
        x = self.SIDEBAR_X + 22
        cv2.putText(canvas, "CONTROLEER DEZE PERSOON", (x, 38), 0, 0.64, (255, 255, 255), 2)
        cv2.putText(
            canvas,
            f"{self._identity_label(track)} | {self.track_index + 1}/{len(self.tracks)}",
            (x, 72), 0, 0.62, (0, 255, 255), 2,
        )
        cv2.putText(
            canvas,
            f"Voorbeeld {self.sample_index + 1}/{len(track.observations)} | frame {frame_number}",
            (x, 102), 0, 0.48, (210, 210, 210), 1,
        )
        automatic = (
            f"{self.team_a_name if track.final_team_id == 0 else self.team_b_name} "
            f"({track.team_agreement_ratio:.0%})"
            if track.final_team_id in (0, 1)
            else "Onbekend"
        )
        cv2.putText(canvas, f"Automatisch: {automatic}", (x, 132), 0, 0.48, (210, 210, 210), 1)
        current = self.corrections.get(track.track_id, track.segment_index)
        current_label = (
            f"{current.role.value} / {current.team.value}"
            if current is not None else "nog niet gecontroleerd"
        )
        cv2.putText(canvas, f"Jouw keuze: {current_label}", (x, 162), 0, 0.45, (120, 255, 120), 1)
        canvas[178:328, x:x + 360] = detail
        cv2.rectangle(canvas, (x, 178), (x + 360, 328), (0, 255, 255), 2)

        self.button_regions = []
        button_y = 342
        for action in ACTIONS:
            region = (x, button_y, self.CANVAS_WIDTH - 22, button_y + 48)
            self.button_regions.append((region, action.key))
            cv2.rectangle(canvas, region[:2], region[2:], action.color, -1)
            cv2.putText(
                canvas,
                self._action_label(action),
                (x + 14, button_y + 32),
                0,
                0.47,
                (255, 255, 255),
                2,
            )
            button_y += 56

        scope_text = (
            "Kies een knop: geldt alleen voor DIT TRACKSEGMENT."
            if track.segment_index is not None
            else "Kies een knop: geldt voor deze VOLLEDIGE persoon."
        )
        cv2.putText(canvas, scope_text, (x, 755), 0, 0.43, (255, 255, 255), 1)
        cv2.putText(canvas, "A/D: ander voorbeeld", (x, 790), 0, 0.43, (200, 200, 200), 1)
        cv2.putText(canvas, "P/N: vorige/volgende persoon", (x, 814), 0, 0.43, (200, 200, 200), 1)
        cv2.putText(canvas, "Q of Esc: opslaan en sluiten", (x, 838), 0, 0.43, (200, 200, 200), 1)
        reviewed = sum(
            self.corrections.get(item.track_id, item.segment_index) is not None
            for item in self.tracks
        )
        cv2.putText(canvas, f"Voortgang: {reviewed}/{len(self.tracks)}", (x, 878), 0, 0.50, (0, 255, 255), 2)

    def _action_label(self, action: ReviewAction) -> str:
        labels = {
            "player_a": f"SPELER {self.team_a_name}",
            "goalkeeper_a": f"KEEPER {self.team_a_name}",
            "player_b": f"SPELER {self.team_b_name}",
            "goalkeeper_b": f"KEEPER {self.team_b_name}",
        }
        return labels.get(action.key, action.label)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (x1, y1, x2, y2), action in self.button_regions:
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._apply(action)
                return

    def _apply(self, action: str) -> None:
        if self.current_track.segment_index is not None:
            correction = correction_for_action(
                self.current_track.track_id,
                action,
                segment_index=self.current_track.segment_index,
            )
            self.corrections = self.corrections.with_correction(correction)
            self._save()
            self._advance_after_review()
            return
        track_ids = self._identity_track_ids(self.current_track)
        for track_id in track_ids:
            correction = correction_for_action(track_id, action)
            self.corrections = self.corrections.with_correction(correction)
        self._save()
        self._advance_after_review()

    def _handle_key(self, key: int) -> None:
        if key in (27, ord("q"), ord("Q")):
            self.running = False
        elif key in (ord("a"), ord("A")):
            self.sample_index = (self.sample_index - 1) % len(self.current_track.observations)
        elif key in (ord("d"), ord("D")):
            self.sample_index = (self.sample_index + 1) % len(self.current_track.observations)
        elif key in (ord("p"), ord("P")):
            self._move_track(-1)
        elif key in (ord("n"), ord("N")):
            self._move_track(1)

    def _move_track(self, step: int) -> None:
        self.track_index = min(
            max(self.track_index + step, 0),
            len(self.tracks) - 1,
        )
        self.sample_index = 0

    def _advance_after_review(self) -> None:
        if self.track_index >= len(self.tracks) - 1:
            print("Review voltooid; alle getoonde items zijn opgeslagen.")
            self.running = False
            return
        self._move_track(1)

    def _save(self) -> None:
        save_entity_corrections(self.corrections, self.output_path)

    def _identity_track_ids(self, track: ReviewTrack) -> tuple[int, ...]:
        if self.identities is None:
            return (track.track_id,)
        identity = self.identities.identity_for_track(track.track_id)
        return identity.track_ids if identity is not None else (track.track_id,)

    def _identity_label(self, track: ReviewTrack) -> str:
        if track.segment_index is not None:
            return f"Tracksegment {track.track_id}.{track.segment_index}"
        if self.identities is None:
            return f"Track ID {track.track_id}"
        identity = self.identities.identity_for_track(track.track_id)
        if identity is None:
            return f"Track ID {track.track_id}"
        return f"{identity.label} ({len(identity.track_ids)} fragment(en))"
