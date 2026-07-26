from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from football_ai.calibration.bootstrap.goal_seed import GoalSeed, load_goal_seeds
from football_ai.calibration.camera_projection_3d import (
    CameraProjection3D,
    CameraProjectionEstimate,
)
from football_ai.calibration.geometry_validation import validate_projected_pitch_geometry
from football_ai.calibration.goal_plane_camera import GoalPlaneCameraConfig, estimate_camera_from_goal_plane
from football_ai.calibration.reference_3d import FootballFieldReference3D
from football_ai.calibration.reference_observation import (
    CameraViewObservations,
    ObservationSource,
    ReferenceObservation2D,
)


@dataclass(frozen=True, slots=True)
class ObservationCollectionResult:
    view: CameraViewObservations
    requested_landmarks: tuple[str, ...]
    skipped_landmarks: tuple[str, ...]
    estimate: CameraProjectionEstimate | None
    failure_reason: str | None

    def to_dict(self) -> dict:
        result = {
            "schema_version": 1,
            "view": self.view.to_dict(),
            "requested_landmarks": list(self.requested_landmarks),
            "skipped_landmarks": list(self.skipped_landmarks),
            "solved": self.estimate is not None,
            "failure_reason": self.failure_reason,
        }
        if self.estimate is not None:
            result["projection"] = {
                "matrix": self.estimate.projection.matrix.tolist(),
                "ground_homography": self.estimate.projection.ground_homography().tolist(),
                "point_errors_px": list(self.estimate.point_errors_px),
                "rms_error_px": self.estimate.rms_error_px,
                "maximum_error_px": self.estimate.maximum_error_px,
            }
        return result


def prefill_goal_observations(seed: GoalSeed) -> tuple[ReferenceObservation2D, ...]:
    goal_id = seed.goal_id.lower()
    if goal_id not in ("a", "b"):
        raise ValueError(f"Onbekend doel: {seed.goal_id}")
    observations = [
        ReferenceObservation2D(f"goal_{goal_id}_rear_bottom", seed.first_ground),
        ReferenceObservation2D(f"goal_{goal_id}_front_bottom", seed.second_ground),
    ]
    if seed.rear_corner is not None:
        observations.append(ReferenceObservation2D(f"corner_{goal_id}_rear", seed.rear_corner))
    if seed.front_corner is not None:
        observations.append(ReferenceObservation2D(f"corner_{goal_id}_front", seed.front_corner))
    return tuple(observations)


def save_observation_result(result: ObservationCollectionResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def orient_projection_toward_field(
    reference: FootballFieldReference3D,
    estimate: CameraProjectionEstimate,
    seed: GoalSeed,
) -> CameraProjectionEstimate:
    """Resolve which side of the observed goal plane contains the pitch.

    Goal-plane observations cannot distinguish the two longitudinal directions.
    Existing sideline support clicks provide that direction without pretending
    they are exact metric landmarks.
    """
    supports = (seed.rear_sideline_support, seed.front_sideline_support)
    if all(item is None for item in supports):
        raise ValueError("Veldrichting ontbreekt: minimaal één bestaand zijlijnpunt vereist.")
    goal = seed.goal_id.lower()
    goal_x = 0.0 if goal == "a" else reference.pitch_length_m
    reflection = np.eye(4, dtype=np.float64)
    reflection[0, 0] = -1.0
    reflection[0, 3] = 2.0 * goal_x
    candidates = (
        estimate.projection,
        CameraProjection3D(estimate.projection.matrix @ reflection),
    )

    chosen = max(candidates, key=lambda item: field_direction_score(reference, item, seed))
    return CameraProjectionEstimate(
        projection=chosen,
        point_errors_px=estimate.point_errors_px,
        rms_error_px=estimate.rms_error_px,
        maximum_error_px=estimate.maximum_error_px,
    )


def field_direction_score(
    reference: FootballFieldReference3D,
    projection: CameraProjection3D,
    seed: GoalSeed,
) -> float:
    goal = seed.goal_id.lower()
    scores: list[float] = []
    for side, support in zip(
        ("rear", "front"),
        (seed.rear_sideline_support, seed.front_sideline_support),
    ):
        if support is None:
            continue
        goal_corner = np.asarray(
            projection.project(reference.landmark(f"corner_{goal}_{side}").point),
            dtype=np.float64,
        )
        other_goal = "b" if goal == "a" else "a"
        other_corner = np.asarray(
            projection.project(reference.landmark(f"corner_{other_goal}_{side}").point),
            dtype=np.float64,
        )
        expected = other_corner - goal_corner
        observed = np.asarray(support, dtype=np.float64) - goal_corner
        denominator = float(np.linalg.norm(expected) * np.linalg.norm(observed))
        if denominator > 1e-6:
            scores.append(float(expected @ observed / denominator))
    return min(scores) if scores else -1.0


class ReferenceObservationApp:
    WINDOW = "Football AI - 3D referentiepunten"

    def __init__(
        self,
        video_path: Path,
        seed: GoalSeed,
        reference: FootballFieldReference3D,
    ) -> None:
        self.video_path = video_path
        self.seed = seed
        self.reference = reference
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Video kon niet worden geopend: {video_path}")
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, seed.frame_number)
        success, frame = self.capture.read()
        self.capture.release()
        if not success:
            raise RuntimeError(f"Frame {seed.frame_number} kon niet worden gelezen.")
        self.frame = frame
        goal = seed.goal_id.lower()
        self.requested = (
            f"goal_{goal}_rear_top",
            f"goal_{goal}_front_top",
        )
        self.observations = list(prefill_goal_observations(seed))
        self.skipped: list[str] = []
        self.index = 0
        self.zoom = 1.0
        self.zoom_center: tuple[float, float] | None = None
        self.image_rect = (430, 0, 1170, 900)
        self.view_origin = (0.0, 0.0)
        self.view_scale = 1.0
        self.status = "Klik het gevraagde punt. Druk S wanneer het echt niet zichtbaar is."

    def run(self) -> ObservationCollectionResult:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 1600, 900)
        cv2.setMouseCallback(self.WINDOW, self._mouse)
        while True:
            cv2.imshow(self.WINDOW, self._render())
            key = cv2.waitKeyEx(30)
            if key == 27:
                cv2.destroyWindow(self.WINDOW)
                raise RuntimeError("Verzamelen van 3D-referentiepunten afgebroken.")
            if key in (ord("s"), ord("S")) and self.index < len(self.requested):
                self.skipped.append(self.requested[self.index])
                self.index += 1
                self.status = "Punt overgeslagen. De software verzint geen ontbrekende positie."
            elif key in (8, 127, ord("u"), ord("U")):
                self._undo()
            elif key in (ord("+"), ord("=")):
                self.zoom = min(8.0, self.zoom * 1.25)
            elif key in (ord("-"), ord("_")):
                self.zoom = max(1.0, self.zoom * 0.8)
            elif key == ord("0"):
                self.zoom, self.zoom_center = 1.0, None
            elif key in (10, 13):
                if self.index < len(self.requested):
                    self.status = "Werk eerst alle vier vragen af; gebruik S als een punt niet zichtbaar is."
                    continue
                cv2.destroyWindow(self.WINDOW)
                return self._build_result()

    def _mouse(self, event: int, x: int, y: int, flags: int, _data: object) -> None:
        if event == cv2.EVENT_MOUSEWHEEL:
            point = self._canvas_to_frame(x, y)
            if point is not None:
                self.zoom_center = point
                delta = np.int16((flags >> 16) & 0xFFFF).item()
                self.zoom = min(8.0, self.zoom * 1.25) if delta > 0 else max(1.0, self.zoom * 0.8)
            return
        if event != cv2.EVENT_LBUTTONDOWN or self.index >= len(self.requested):
            return
        point = self._canvas_to_frame(x, y)
        if point is None:
            self.status = "Klik binnen het videobeeld, niet in het instructiepaneel."
            return
        landmark_id = self.requested[self.index]
        self.observations.append(ReferenceObservation2D(landmark_id, point))
        self.index += 1
        self.status = "Punt opgeslagen. Controleer het magenta kruis; U maakt de laatste stap ongedaan."

    def _undo(self) -> None:
        if self.index == 0:
            self.status = "Er is nog geen nieuw punt om ongedaan te maken."
            return
        self.index -= 1
        landmark_id = self.requested[self.index]
        self.skipped = [item for item in self.skipped if item != landmark_id]
        self.observations = [item for item in self.observations if item.landmark_id != landmark_id]
        self.status = "Laatste stap ongedaan gemaakt."

    def _build_result(self) -> ObservationCollectionResult:
        view = CameraViewObservations(
            frame_number=self.seed.frame_number,
            camera_state=self.seed.camera_state,
            observations=tuple(self.observations),
        )
        estimate: CameraProjectionEstimate | None = None
        failure: str | None = None
        try:
            candidates: list[CameraProjectionEstimate] = []
            for horizontal_fov in np.linspace(35.0, 115.0, 17):
                try:
                    candidate = estimate_camera_from_goal_plane(
                        self.reference,
                        view,
                        frame_size=(self.frame.shape[1], self.frame.shape[0]),
                        config=GoalPlaneCameraConfig(
                            horizontal_fov_degrees=float(horizontal_fov),
                            use_plane_focal_estimate=False,
                        ),
                    )
                    candidates.append(orient_projection_toward_field(self.reference, candidate, self.seed))
                except ValueError:
                    continue
            if not candidates:
                raise ValueError("Geen fysiek bruikbaar cameramodel gevonden.")
            estimate = max(
                candidates,
                key=lambda item: field_direction_score(self.reference, item.projection, self.seed),
            )
            direction_score = field_direction_score(self.reference, estimate.projection, self.seed)
            if direction_score < 0.75:
                raise ValueError("Cameramodel volgt de twee bestaande zijlijnrichtingen onvoldoende.")
            field_ids = ("corner_a_rear", "corner_b_rear", "corner_b_front", "corner_a_front")
            field = np.asarray(
                [estimate.projection.project(self.reference.landmark(item).point) for item in field_ids],
                dtype=np.float64,
            )
            geometry = validate_projected_pitch_geometry(
                field,
                frame_width=self.frame.shape[1],
                frame_height=self.frame.shape[0],
            )
            quality_errors = list(geometry.errors)
            if estimate.rms_error_px > 8.0:
                quality_errors.append(f"Aanklikfout is te groot (RMS {estimate.rms_error_px:.1f} px).")
            if estimate.maximum_error_px > 20.0:
                quality_errors.append(f"Eén aanklikfout is te groot (max {estimate.maximum_error_px:.1f} px).")
            if quality_errors:
                estimate = None
                failure = " ".join(quality_errors)
        except ValueError as error:
            failure = str(error)
        return ObservationCollectionResult(view, self.requested, tuple(self.skipped), estimate, failure)

    def _render(self) -> np.ndarray:
        canvas = np.full((900, 1600, 3), 24, np.uint8)
        lines = [
            "3D REFERENTIE - EEN CAMERABEELD",
            f"Doel {self.seed.goal_id} | frame {self.seed.frame_number} | {self.seed.time_seconds:.1f}s",
            "",
            "Al bekend en zichtbaar in magenta:",
            "- beide paalvoeten",
            "- beide hoeken van deze achterlijn",
            "",
            f"Stap {min(self.index + 1, len(self.requested))}/{len(self.requested)}",
            self._instruction(),
            "",
            "Klik exact in het VIDEOBEELD.",
            "S = niet zichtbaar / overslaan",
            "U = laatste stap ongedaan",
            "Muiswiel of +/- = inzoomen",
            "0 = volledig beeld",
            "Enter = afronden na alle vragen",
            "Esc = afbreken",
            "",
            "Belangrijk:",
            "Een overgeslagen punt wordt niet geschat.",
            "Zonder beide paaltoppen volgt geen veldvak.",
        ]
        for index, line in enumerate(lines):
            color = (0, 230, 255) if index in (0, 7, 8) else (235, 235, 235)
            cv2.putText(canvas, line, (18, 34 + index * 31), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
        for index, line in enumerate(self._wrap(self.status, 46)):
            cv2.putText(canvas, line, (18, 820 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 255), 1, cv2.LINE_AA)
        self._draw_frame(canvas)
        return canvas

    def _instruction(self) -> str:
        if self.index >= len(self.requested):
            return "KLAAR: druk Enter om de geometrie te controleren."
        goal = self.seed.goal_id
        return {
            f"goal_{goal.lower()}_rear_top": "Klik BOVENKANT van de VERSTE doelpaal.",
            f"goal_{goal.lower()}_front_top": "Klik BOVENKANT van de DICHTSTBIJZIJNDE doelpaal.",
        }[self.requested[self.index]]

    def _draw_frame(self, canvas: np.ndarray) -> None:
        annotated = self.frame.copy()
        for observation in self.observations:
            center = tuple(np.round(observation.image_point).astype(int))
            cv2.drawMarker(annotated, center, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 3, cv2.LINE_AA)
            cv2.putText(annotated, observation.landmark_id, (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1, cv2.LINE_AA)
        height, width = annotated.shape[:2]
        view_width, view_height = width / self.zoom, height / self.zoom
        center = self.zoom_center or (width / 2.0, height / 2.0)
        origin_x = min(max(0.0, center[0] - view_width / 2.0), width - view_width)
        origin_y = min(max(0.0, center[1] - view_height / 2.0), height - view_height)
        crop = annotated[int(origin_y):int(origin_y + view_height), int(origin_x):int(origin_x + view_width)]
        scale = min(1170 / crop.shape[1], 900 / crop.shape[0])
        shown = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
        x, y = 430, (900 - shown.shape[0]) // 2
        canvas[y:y + shown.shape[0], x:x + shown.shape[1]] = shown
        self.image_rect = (x, y, shown.shape[1], shown.shape[0])
        self.view_origin, self.view_scale = (origin_x, origin_y), scale

    def _canvas_to_frame(self, x: int, y: int) -> tuple[float, float] | None:
        rx, ry, rw, rh = self.image_rect
        if not (rx <= x < rx + rw and ry <= y < ry + rh):
            return None
        return self.view_origin[0] + (x - rx) / self.view_scale, self.view_origin[1] + (y - ry) / self.view_scale

    @staticmethod
    def _wrap(text: str, width: int) -> list[str]:
        lines: list[str] = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if len(candidate) > width and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines[-3:]


def load_goal_seed(path: Path, goal_id: str) -> GoalSeed:
    goal_id = goal_id.upper()
    for seed in load_goal_seeds(path):
        if seed.goal_id.upper() == goal_id:
            return seed
    raise ValueError(f"Doel {goal_id} ontbreekt in {path}.")


def create_projection_preview(
    frame: np.ndarray,
    reference: FootballFieldReference3D,
    result: ObservationCollectionResult,
) -> np.ndarray:
    preview = frame.copy()
    if result.estimate is None:
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 70), (20, 20, 20), -1)
        cv2.putText(preview, f"GEEN OPLOSSING: {result.failure_reason}", (15, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2, cv2.LINE_AA)
        return preview
    projection = result.estimate.projection
    field_ids = ("corner_a_rear", "corner_b_rear", "corner_b_front", "corner_a_front")
    field = np.asarray([projection.project(reference.landmark(item).point) for item in field_ids], dtype=np.int32)
    cv2.polylines(preview, [field], True, (0, 255, 255), 4, cv2.LINE_AA)
    goal = result.view.observations[0].landmark_id.split("_")[1]
    goal_ids = (
        f"goal_{goal}_rear_bottom", f"goal_{goal}_rear_top",
        f"goal_{goal}_front_top", f"goal_{goal}_front_bottom",
    )
    goal_box = np.asarray([projection.project(reference.landmark(item).point) for item in goal_ids], dtype=np.int32)
    cv2.polylines(preview, [goal_box], True, (255, 255, 0), 3, cv2.LINE_AA)
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 64), (20, 20, 20), -1)
    cv2.putText(preview, f"3D QA | RMS {result.estimate.rms_error_px:.2f}px | max {result.estimate.maximum_error_px:.2f}px", (15, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return preview
