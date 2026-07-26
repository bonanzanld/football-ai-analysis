from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from football_ai.calibration.bootstrap.detection_profile import PitchDetectionProfile
from football_ai.calibration.bootstrap.goal_detection import (
    GoalDetection,
    detect_goal_candidates,
    draw_goal_detection,
)
from football_ai.calibration.bootstrap.temporal_goal_confirmation import (
    ConfirmedGoal,
    confirm_goals_temporally,
)
from football_ai.calibration.bootstrap.goal_pair_selection import (
    CameraStateGoalEvidence,
    GoalPairSelection,
    select_opposing_goal_pair,
)
from football_ai.calibration.bootstrap.white_line_detection import (
    WhiteLineDetection,
    detect_white_field_lines,
    draw_white_line_detection,
)


@dataclass(frozen=True, slots=True)
class CameraStateLineDetection:
    camera_state: int
    frame_number: int
    time_seconds: float
    stable: bool
    view_position: float
    frame: np.ndarray
    detection: WhiteLineDetection
    goal_detection: GoalDetection
    confirmed_goals: tuple[ConfirmedGoal, ...]


class BootstrapLineDetectionAnalyzer:
    def __init__(self, profile: PitchDetectionProfile) -> None:
        self.profile = profile

    def analyze(
        self,
        video_path: Path,
        bootstrap_report_path: Path,
    ) -> tuple[CameraStateLineDetection, ...]:
        bootstrap = json.loads(bootstrap_report_path.read_text(encoding="utf-8"))
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Video kon niet worden geopend: {video_path}")
        results: list[CameraStateLineDetection] = []
        samples_by_index = {
            int(item["sample_index"]): item for item in bootstrap["samples"]
        }
        for state in bootstrap["camera_states"]:
            frame_number = int(state["representative_frame_number"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = capture.read()
            if not success:
                continue
            state_sample_indices = [
                int(item["sample_index"])
                for item in bootstrap["samples"]
                if int(item["camera_state"]) == int(state["camera_state"])
            ]
            selected_indices = _evenly_spaced(state_sample_indices, maximum_count=12)
            temporal_detections: list[GoalDetection] = []
            temporal_sizes: list[tuple[int, int]] = []
            for sample_index in selected_indices:
                sample = samples_by_index[sample_index]
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(sample["frame_number"]))
                temporal_success, temporal_frame = capture.read()
                if not temporal_success:
                    continue
                temporal_detections.append(detect_goal_candidates(temporal_frame))
                temporal_sizes.append((temporal_frame.shape[1], temporal_frame.shape[0]))
            confirmed_goals = confirm_goals_temporally(
                temporal_detections,
                temporal_sizes,
            )
            results.append(
                CameraStateLineDetection(
                    camera_state=int(state["camera_state"]),
                    frame_number=frame_number,
                    time_seconds=float(state["representative_time_seconds"]),
                    stable=bool(state["stable"]),
                    view_position=float(state.get("view_position", 0.5)),
                    frame=frame,
                    detection=detect_white_field_lines(frame, self.profile),
                    goal_detection=detect_goal_candidates(frame),
                    confirmed_goals=confirmed_goals,
                )
            )
        capture.release()
        return tuple(results)

    def save_json(
        self,
        results: tuple[CameraStateLineDetection, ...],
        path: Path,
    ) -> None:
        pair_selection = self.select_goal_pair(results)
        data = {
            "schema_version": 1,
            "match_format": self.profile.match_format.value,
            "profile": {
                "name": self.profile.name,
                "pitch_length_m": self.profile.pitch_length_m,
                "pitch_width_m": self.profile.pitch_width_m,
                "goal_width_m": self.profile.goal_width_m,
                "white_line_evidence_weight": self.profile.white_line_evidence_weight,
                "boundary_marker_evidence_weight": self.profile.boundary_marker_evidence_weight,
                "goal_evidence_weight": self.profile.goal_evidence_weight,
                "notes": list(self.profile.notes),
            },
            "camera_states": [
                {
                    "camera_state": item.camera_state,
                    "frame_number": item.frame_number,
                    "time_seconds": item.time_seconds,
                    "stable": item.stable,
                    "grass_coverage": item.detection.grass_coverage,
                    "white_pixel_ratio": item.detection.white_pixel_ratio,
                    "line_candidate_count": len(item.detection.candidates),
                    "line_candidates": [candidate.to_dict() for candidate in item.detection.candidates],
                    "goal_candidate_count": len(item.goal_detection.candidates),
                    "goal_candidates": [candidate.to_dict() for candidate in item.goal_detection.candidates],
                    "confirmed_goal_count": len(item.confirmed_goals),
                    "confirmed_goals": [goal.to_dict() for goal in item.confirmed_goals],
                }
                for item in results
            ],
            "opposing_goal_pair": (
                pair_selection.pair.to_dict() if pair_selection.pair is not None else None
            ),
            "goal_pair_reason": pair_selection.reason,
            "evaluated_goal_pair_count": pair_selection.evaluated_pair_count,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def create_contact_sheet(
        self,
        results: tuple[CameraStateLineDetection, ...],
    ) -> np.ndarray:
        tile_width, tile_height = 480, 270
        columns = min(3, len(results))
        rows = int(np.ceil(len(results) / columns))
        header_height = 78
        sheet = np.zeros((header_height + rows * tile_height, columns * tile_width, 3), np.uint8)
        pair_selection = self.select_goal_pair(results)
        _text(sheet, f"WITTE LIJNEN - {self.profile.name.upper()}", (16, 30), 0.72, 2)
        _text(
            sheet,
            pair_selection.reason,
            (16, 59),
            0.52,
            1,
        )
        for index, item in enumerate(results):
            if self.profile.match_format.value == "8v8":
                confirmed_detection = GoalDetection(
                    tuple(goal.representative for goal in item.confirmed_goals)
                )
                overlay = draw_goal_detection(item.frame, confirmed_detection)
                if item.confirmed_goals:
                    _draw_backline_hypothesis(
                        overlay,
                        item.confirmed_goals[0].representative,
                    )
            else:
                overlay = draw_white_line_detection(item.frame, item.detection)
            overlay = cv2.resize(overlay, (tile_width, tile_height))
            cv2.rectangle(overlay, (0, 0), (tile_width, 42), (18, 18, 18), -1)
            _text(
                overlay,
                f"Stand {item.camera_state} | {len(item.confirmed_goals)} bevestigd | "
                f"{item.time_seconds:.1f}s",
                (12, 28),
                0.55,
                2,
            )
            row, column = divmod(index, columns)
            y, x = header_height + row * tile_height, column * tile_width
            sheet[y:y + tile_height, x:x + tile_width] = overlay
        return sheet

    @staticmethod
    def select_goal_pair(
        results: tuple[CameraStateLineDetection, ...],
    ) -> GoalPairSelection:
        return select_opposing_goal_pair(
            tuple(
                CameraStateGoalEvidence(
                    camera_state=item.camera_state,
                    view_position=item.view_position,
                    frame_width=item.frame.shape[1],
                    frame_height=item.frame.shape[0],
                    confirmed_goals=item.confirmed_goals,
                )
                for item in results
            )
        )


def _text(image: np.ndarray, value: str, origin: tuple[int, int], scale: float, thickness: int) -> None:
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )


def _draw_backline_hypothesis(image: np.ndarray, goal) -> None:
    first = np.asarray(goal.left_ground, dtype=np.float64)
    second = np.asarray(goal.right_ground, dtype=np.float64)
    direction = second - first
    if abs(direction[0]) >= abs(direction[1]) and abs(direction[0]) > 1e-6:
        left_y = first[1] + (0.0 - first[0]) * direction[1] / direction[0]
        right_x = image.shape[1] - 1.0
        right_y = first[1] + (right_x - first[0]) * direction[1] / direction[0]
        start, end = (0, int(round(left_y))), (int(right_x), int(round(right_y)))
    elif abs(direction[1]) > 1e-6:
        top_x = first[0] + (0.0 - first[1]) * direction[0] / direction[1]
        bottom_y = image.shape[0] - 1.0
        bottom_x = first[0] + (bottom_y - first[1]) * direction[0] / direction[1]
        start, end = (int(round(top_x)), 0), (int(round(bottom_x)), int(bottom_y))
    else:
        return
    cv2.line(image, start, end, (0, 255, 255), 3, cv2.LINE_AA)


def _evenly_spaced(indices: list[int], maximum_count: int) -> list[int]:
    if len(indices) <= maximum_count:
        return indices
    positions = np.linspace(0, len(indices) - 1, maximum_count)
    return [indices[int(round(position))] for position in positions]
