from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from football_ai.calibration.camera_anchor_bank_3d import load_camera_anchor_bank
from football_ai.calibration.camera_profile import CameraKind, ZoomMode, create_camera_profile
from football_ai.calibration.lens_geometry import StraightLineObservation, estimate_radial_distortion_from_lines
from football_ai.calibration.manual_perspective_reference import (
    ManualPerspectiveReference,
    ManualPerspectiveView,
    ManualReferenceLine,
    PerspectiveDirection,
    automatically_classify_line_directions,
    assess_three_view_consistency,
    assess_global_readiness,
    draw_manual_perspective_view,
    save_manual_perspective_reference,
)


class PerspectiveCollector:
    WINDOW = "Handmatige perspectiefreferentie"
    PANEL = 500
    VIEW_W = 1400
    VIEW_H = 820
    POINTS_PER_LINE = 5

    def __init__(self, video: Path, initial_views: tuple[tuple[str, float], ...]) -> None:
        self.video = video
        self.capture = cv2.VideoCapture(str(video))
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Video kon niet worden geopend: {video}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.initial_views = initial_views
        self.view_index = 0
        self.time_seconds = initial_views[0][1]
        self.frame = self._read(self.time_seconds)
        self.zoom = 1.0
        self.zoom_center: tuple[float, float] | None = None
        self.lines: list[ManualReferenceLine] = []
        self.points: list[tuple[float, float]] = []
        self.views: list[ManualPerspectiveView] = []
        self.status = "Kies zo nodig een beter frame; klik daarna de gevraagde witte lijn."
        self.requests = tuple(range(1, 7))

    def run(self) -> tuple[ManualPerspectiveView, ...]:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self.PANEL + self.VIEW_W, self.VIEW_H)
        cv2.setMouseCallback(self.WINDOW, self._mouse)
        try:
            while self.view_index < len(self.initial_views):
                cv2.imshow(self.WINDOW, self._render())
                key = cv2.waitKeyEx(30)
                if key in (27, ord("q"), ord("Q")):
                    raise KeyboardInterrupt("Perspectiefinvoer afgebroken.")
                if key in (ord("u"), ord("U")):
                    self._undo()
                elif key in (ord("+"), ord("=")):
                    self.zoom = min(8.0, self.zoom * 1.25)
                elif key in (ord("-"), ord("_")):
                    self.zoom = max(1.0, self.zoom / 1.25)
                elif key == ord("0"):
                    self.zoom, self.zoom_center = 1.0, None
                elif key in (ord(","), 2424832):
                    self._shift_frame(-1.0)
                elif key in (ord("."), 2555904):
                    self._shift_frame(1.0)
                elif key in (10, 13) and len(self.lines) >= 2 and not self.points:
                    self._finish_view()
        finally:
            self.capture.release()
            cv2.destroyWindow(self.WINDOW)
        return tuple(self.views)

    def _read(self, time_seconds: float) -> np.ndarray:
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(time_seconds * self.fps)))
        success, frame = self.capture.read()
        if not success:
            raise RuntimeError(f"Frame rond {time_seconds:.1f}s kon niet worden gelezen.")
        return frame

    def _shift_frame(self, seconds: float) -> None:
        if self.lines or self.points:
            self.status = "Wis eerst de klikken met U voordat je een ander frame kiest."
            return
        self.time_seconds = max(0.0, self.time_seconds + seconds)
        self.frame = self._read(self.time_seconds)
        self.zoom, self.zoom_center = 1.0, None

    def _mouse(self, event: int, x: int, y: int, flags: int, _data) -> None:
        if x < self.PANEL:
            return
        if event == cv2.EVENT_MOUSEWHEEL:
            point = self._display_to_image(x - self.PANEL, y)
            if point is not None:
                self.zoom_center = point
                self.zoom = min(8.0, self.zoom * 1.25) if flags > 0 else max(1.0, self.zoom / 1.25)
            return
        if event != cv2.EVENT_LBUTTONDOWN or len(self.lines) >= len(self.requests):
            return
        point = self._display_to_image(x - self.PANEL, y)
        if point is None:
            return
        self.points.append(point)
        if len(self.points) == self.POINTS_PER_LINE:
            self.lines.append(ManualReferenceLine(PerspectiveDirection.UNKNOWN, tuple(self.points)))
            self.points.clear()
            self.status = "Lijn opgeslagen. Controleer de gekleurde lijn; U maakt hem ongedaan."

    def _undo(self) -> None:
        if self.points:
            self.points.pop()
        elif self.lines:
            line = self.lines.pop()
            self.points = list(line.points)
            self.points.pop()
        else:
            self.status = "Er is niets om ongedaan te maken."

    def _finish_view(self) -> None:
        label = self.initial_views[self.view_index][0]
        frame_number = int(round(self.time_seconds * self.fps))
        if len(self.lines) >= 4:
            try:
                lines = automatically_classify_line_directions(
                    tuple(self.lines),
                    (self.frame.shape[1], self.frame.shape[0]),
                )
            except ValueError as error:
                self.status = f"KAN NIET AFRONDEN: {error} Gebruik U om lijnen opnieuw te kiezen."
                return
        else:
            lines = tuple(self.lines)
        view = ManualPerspectiveView(label, frame_number, self.time_seconds, lines)
        self.views.append(view)
        self.view_index += 1
        if self.view_index >= len(self.initial_views):
            return
        self.time_seconds = self.initial_views[self.view_index][1]
        self.frame = self._read(self.time_seconds)
        self.lines, self.points = [], []
        self.zoom, self.zoom_center = 1.0, None
        self.status = "Nieuw camerabeeld. Kies zo nodig met , en . een beter frame."

    def _crop(self) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        height, width = self.frame.shape[:2]
        crop_w, crop_h = width / self.zoom, height / self.zoom
        cx, cy = self.zoom_center or (width / 2.0, height / 2.0)
        x0 = float(np.clip(cx - crop_w / 2.0, 0.0, width - crop_w))
        y0 = float(np.clip(cy - crop_h / 2.0, 0.0, height - crop_h))
        crop = self.frame[int(y0):int(y0 + crop_h), int(x0):int(x0 + crop_w)]
        return crop, (x0, y0, crop_w, crop_h)

    def _display_to_image(self, x: int, y: int) -> tuple[float, float] | None:
        _crop, (x0, y0, crop_w, crop_h) = self._crop()
        scale = min(self.VIEW_W / crop_w, self.VIEW_H / crop_h)
        shown_w, shown_h = crop_w * scale, crop_h * scale
        offset_x, offset_y = (self.VIEW_W - shown_w) / 2.0, (self.VIEW_H - shown_h) / 2.0
        if not (offset_x <= x <= offset_x + shown_w and offset_y <= y <= offset_y + shown_h):
            return None
        if y < offset_y + 126:
            self.status = "Klik onder de zwarte instructiebalk, rechtstreeks op de witte kalklijn."
            return None
        return x0 + (x - offset_x) / scale, y0 + (y - offset_y) / scale

    def _render(self) -> np.ndarray:
        canvas = np.full((self.VIEW_H, self.PANEL + self.VIEW_W, 3), 24, dtype=np.uint8)
        crop, (x0, y0, crop_w, crop_h) = self._crop()
        scale = min(self.VIEW_W / crop_w, self.VIEW_H / crop_h)
        resized = cv2.resize(crop, (int(round(crop_w * scale)), int(round(crop_h * scale))))
        ox = self.PANEL + (self.VIEW_W - resized.shape[1]) // 2
        oy = (self.VIEW_H - resized.shape[0]) // 2
        canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
        colors = {
            PerspectiveDirection.BETWEEN_GOALS: (0, 255, 255),
            PerspectiveDirection.ALONG_END_LINES: (255, 255, 0),
            PerspectiveDirection.UNKNOWN: (0, 165, 255),
        }
        for line in self.lines:
            pts = [self._image_to_display(item, x0, y0, scale, ox, oy) for item in line.points]
            fitted = line.endpoints(self.frame.shape[1], self.frame.shape[0])
            a, b = [self._image_to_display(item, x0, y0, scale, ox, oy) for item in fitted]
            cv2.line(canvas, a, b, colors[line.direction], 3, cv2.LINE_AA)
            for point in pts:
                cv2.circle(canvas, point, 6, (255, 0, 255), -1, cv2.LINE_AA)
        for item in self.points:
            cv2.circle(canvas, self._image_to_display(item, x0, y0, scale, ox, oy), 6, (255, 0, 255), -1, cv2.LINE_AA)
        self._draw_image_instruction(canvas, ox, oy, resized.shape[1])
        self._draw_panel(canvas)
        return canvas

    @staticmethod
    def _image_to_display(point, x0, y0, scale, ox, oy):
        return int(round(ox + (point[0] - x0) * scale)), int(round(oy + (point[1] - y0) * scale))

    def _draw_panel(self, canvas: np.ndarray) -> None:
        labels = ("LINKERDOEL / DOEL A", "MIDDEN VAN HET VELD", "RECHTERDOEL / DOEL B")
        title, instruction, example, warning, _color = self._current_instruction()
        lines = [
            "HANDMATIGE PERSPECTIEFREFERENTIE",
            f"Beeld {self.view_index + 1}/3: {labels[self.view_index]}",
            f"Tijd {self.time_seconds:.1f}s | zoom {self.zoom:.1f}x",
            "",
            title,
            instruction,
            example,
            warning,
            "",
            "Klikvolgorde op die ene lijn:",
            "Klik 5 punten verspreid van begin tot einde.",
            "Klik midden op de kalkstreep; een kleine marge is goed.",
            "",
            f"Opgeslagen lijnen: {len(self.lines)}/6",
            f"Punten huidige lijn: {len(self.points)}/{self.POINTS_PER_LINE}",
            "",
            ", / . of pijlen: 1 seconde ander frame",
            "Muiswiel of +/-: zoom | 0: volledig beeld",
            "U: laatste klik/lijn ongedaan",
            "Enter: afronden vanaf 2 lijnen | Esc: afbreken",
            "",
            self.status,
        ]
        y = 30
        for index, text in enumerate(lines):
            color = _color if index == 4 else ((0, 255, 255) if index == 0 else (230, 230, 230))
            cv2.putText(canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1 if index else 2, cv2.LINE_AA)
            y += 29

    def _current_instruction(self) -> tuple[str, str, str, str, tuple[int, int, int]]:
        if len(self.lines) >= len(self.requests):
            return (
                "KLAAR MET DIT BEELD",
                "De software verdeelt de lijnen nu automatisch in 2 groepen.",
                "Goed? Druk Enter voor het volgende beeld.",
                "Niet goed? Druk U en klik de lijn opnieuw.",
                (0, 220, 0),
            )
        number = self.requests[len(self.lines)]
        optional = (
            "Je hebt genoeg voor PARTIAL: Enter mag nu, of voeg nog een andere lijn toe."
            if len(self.lines) >= 2
            else "Minimaal 2 verschillende lijnen nodig; meer lijnen maken de oplossing sterker."
        )
        return (
            f"WITTE REFERENTIELIJN {number} VAN 6",
            "Kies zelf EEN goed zichtbare, RECHTE witte 11v11-lijn.",
            "Klik 5 verspreide punten op precies diezelfde kalklijn.",
            optional,
            (0, 165, 255),
        )

    def _draw_image_instruction(self, canvas: np.ndarray, x: int, y: int, width: int) -> None:
        title, instruction, example, warning, color = self._current_instruction()
        banner_height = 126
        cv2.rectangle(canvas, (x, y), (x + width, y + banner_height), (10, 10, 10), -1)
        cv2.putText(canvas, title, (x + 18, y + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, instruction, (x + 18, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, example, (x + 18, y + 89), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, warning, (x + 18, y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verzamel drie handmatige perspectiefreferenties.")
    parser.add_argument("--video", default="Test2_4k_opbouw_keeper.mp4")
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--camera", choices=tuple(item.value for item in CameraKind), default="unknown")
    parser.add_argument("--zoom-mode", choices=tuple(item.value for item in ZoomMode), default="unknown")
    parser.add_argument("--times", help="Tijden links,midden,rechts, bijvoorbeeld 105,5,72.")
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output_dir = PROJECT_ROOT / "output" / "pitch_bootstrap"
    prefix = f"{video.stem}_{args.format}"
    initial_views = _initial_view_times(
        video, args.times, output_dir / f"{prefix}_camera_anchors_3d.json"
    )
    collector = PerspectiveCollector(
        video,
        initial_views,
    )
    views = collector.run()
    reference = ManualPerspectiveReference(video.name, views)
    output = output_dir / f"{prefix}_manual_perspective_reference.json"
    save_manual_perspective_reference(reference, output)
    previews = []
    capture = cv2.VideoCapture(str(video))
    for view in views:
        capture.set(cv2.CAP_PROP_POS_FRAMES, view.frame_number)
        success, frame = capture.read()
        if success:
            preview = draw_manual_perspective_view(frame, view)
            previews.append(cv2.resize(preview, (640, 360)))
    capture.release()
    preview_path = output_dir / f"{prefix}_manual_perspective_reference.jpg"
    if previews:
        cv2.imwrite(str(preview_path), np.hstack(previews))
    frame_size = (collector.frame.shape[1], collector.frame.shape[0])
    profile = create_camera_profile(args.camera, zoom_mode=args.zoom_mode)
    initial_focal = None
    if profile.horizontal_fov_degrees is not None:
        initial_focal = frame_size[0] / (
            2.0 * np.tan(np.deg2rad(profile.horizontal_fov_degrees) / 2.0)
        )
    line_observations = tuple(
        StraightLineObservation(np.asarray(line.points, dtype=np.float64))
        for view in views
        for line in view.lines
    )
    lens = estimate_radial_distortion_from_lines(
        frame_size, line_observations, initial_focal_length_px=initial_focal
    )
    leave_one_view_out = []
    for omitted_index in range(len(views)):
        remaining = tuple(
            StraightLineObservation(np.asarray(line.points, dtype=np.float64))
            for view_index, view in enumerate(views)
            if view_index != omitted_index
            for line in view.lines
        )
        estimate = estimate_radial_distortion_from_lines(
            frame_size, remaining, initial_focal_length_px=initial_focal
        )
        leave_one_view_out.append(estimate.intrinsics.radial_distortion)
    coefficient_samples = np.asarray(leave_one_view_out, dtype=np.float64)
    coefficient_spread = np.ptp(coefficient_samples, axis=0)
    lens_stable = bool(coefficient_spread[0] <= 0.03 and coefficient_spread[1] <= 0.03)
    lens_path = output_dir / f"{prefix}_lens_geometry_qa.json"
    import json
    lens_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "video_name": video.name,
                "camera_profile": profile.kind.value,
                "camera_profile_is_prior_only": True,
                "zoom_mode": profile.zoom_mode.value,
                "frame_size": list(frame_size),
                "focal_length_px": lens.intrinsics.focal_length_px,
                "principal_point": list(lens.intrinsics.principal_point),
                "radial_distortion": list(lens.intrinsics.radial_distortion),
                "line_count": lens.line_count,
                "rms_straightness_px": lens.rms_straightness_px,
                "maximum_straightness_px": lens.maximum_straightness_px,
                "leave_one_view_out": [list(item) for item in leave_one_view_out],
                "coefficient_spread": coefficient_spread.tolist(),
                "stable": lens_stable,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    valid, focals, quality_reason = assess_three_view_consistency(views, frame_size)
    print(f"Perspectiefreferentie opgeslagen: {output}")
    print(f"QA-preview: {preview_path}")
    complete_views = tuple(view for view in views if view.perspective_complete)
    if focals:
        print("Brandpuntschattingen: " + " | ".join(f"{view.label} {focal:.0f}px" for view, focal in zip(complete_views, focals)))
    partial = any(not view.perspective_complete for view in views)
    global_ready, global_reason = assess_global_readiness(views)
    if partial and global_ready:
        status = "READY_FOR_GLOBAL_SOLVE"
        quality_reason = global_reason
    elif valid:
        status = "PARTIAL" if partial else "PASS"
    else:
        status = "PARTIAL" if quality_reason.startswith("PARTIAL:") else "FAIL"
    print(f"Perspectief-QA: {status} | {quality_reason}")
    print(
        f"Lens-QA: {'PASS' if lens_stable else 'FAIL'} | {lens.line_count} lijnen | "
        f"k1 {lens.intrinsics.radial_distortion[0]:.5f} | "
        f"k2 {lens.intrinsics.radial_distortion[1]:.5f} | RMS {lens.rms_straightness_px:.2f}px"
    )
    print(
        f"Lensstabiliteit: spreiding k1 {coefficient_spread[0]:.5f} | "
        f"k2 {coefficient_spread[1]:.5f} (maximaal 0.03000)"
    )
    print(f"Lensrapport: {lens_path}")


def _initial_view_times(
    video: Path,
    requested: str | None,
    anchor_bank_path: Path,
) -> tuple[tuple[str, float], ...]:
    labels = ("left_goal", "center", "right_goal")
    if requested:
        values = tuple(float(item.strip()) for item in requested.split(","))
        if len(values) != 3 or any(value < 0.0 for value in values):
            raise ValueError("--times vereist drie niet-negatieve tijden: links,midden,rechts.")
        return tuple(zip(labels, values))
    if anchor_bank_path.exists():
        bank = load_camera_anchor_bank(anchor_bank_path)
        goal_a = next(item for item in bank.anchors if item.anchor_id == "goal-a")
        goal_b = next(item for item in bank.anchors if item.anchor_id == "goal-b")
        return (
            (labels[0], goal_a.time_seconds),
            (labels[1], 0.5 * (goal_a.time_seconds + goal_b.time_seconds)),
            (labels[2], goal_b.time_seconds),
        )
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    duration = float(capture.get(cv2.CAP_PROP_FRAME_COUNT)) / max(
        float(capture.get(cv2.CAP_PROP_FPS)), 1e-9
    )
    capture.release()
    return ((labels[0], 0.82 * duration), (labels[1], 0.05 * duration), (labels[2], 0.58 * duration))


if __name__ == "__main__":
    main()
