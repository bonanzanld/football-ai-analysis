from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.detection.ball_review import (
    centered_image_box,
    confirm_ball_annotation,
    image_box_from_drag,
    next_required_review_index,
    propagate_ball_box_optical_flow,
    proposed_candidate_box,
    review_progress,
    save_review_manifest,
    visible_human_review_indices,
)


def propagate_from_last_reviewed(payload: dict[str, object], manifest_path: Path) -> int:
    """Create AI drafts after the last reviewed visible anchor."""

    annotations: list[dict[str, object]] = payload["annotations"]
    seeds = [
        index
        for index, annotation in enumerate(annotations)
        if annotation.get("review_status") == "human_reviewed"
        and annotation.get("visibility") == "visible"
        and annotation.get("ball_box") is not None
    ]
    if not seeds:
        raise ValueError("Geen menselijk beoordeeld zichtbaar startanker gevonden")
    seed_index = seeds[-1]
    images = [
        cv2.imread(str(manifest_path.parent / str(annotation["image"])))
        for annotation in annotations
    ]
    if any(image is None for image in images):
        raise FileNotFoundError("Een of meer reviewframes ontbreken")
    boxes = propagate_ball_box_optical_flow(
        images,
        seed_index=seed_index,
        seed_box=annotations[seed_index]["ball_box"],
    )
    drafted = 0
    drafted_indices: list[int] = []
    for index in range(seed_index + 1, len(annotations)):
        box = boxes[index]
        if box is None:
            break
        if annotations[index].get("review_status") == "human_reviewed":
            continue
        annotations[index] = {
            **annotations[index],
            "visibility": "visible",
            "ball_box": list(box),
            "occlusion": "none",
            "review_status": "ai_draft",
            "review_priority": "optional",
            "notes": "Optical-flowvoorstel vanaf menselijk startanker; niet handmatig bevestigd.",
        }
        drafted += 1
        drafted_indices.append(index)
    if drafted_indices:
        annotations[drafted_indices[-1]]["review_priority"] = "required"
        annotations[drafted_indices[-1]]["notes"] = (
            "Eindanker van optical-flowtraject; menselijke controle vereist."
        )
    save_review_manifest(payload, manifest_path)
    return drafted


class BallGroundTruthReviewApp:
    WINDOW = "Football AI - actieve bal annoteren"
    WIDTH = 1600
    HEIGHT = 900
    IMAGE_WIDTH = 1220

    def __init__(
        self,
        manifest_path: Path,
        candidate_cache_path: Path | None = None,
        review_indices: list[int] | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.annotations: list[dict[str, object]] = self.payload["annotations"]
        if not self.annotations:
            raise ValueError("Geen annotatieframes gevonden")
        self.candidate_frames: list[dict[str, object]] = []
        if candidate_cache_path is not None:
            candidate_payload = json.loads(
                candidate_cache_path.read_text(encoding="utf-8")
            )
            frames = candidate_payload.get("frames")
            if candidate_payload.get("schema_version") not in (1, 2) or not isinstance(
                frames, list
            ):
                raise ValueError("Ongeldige kandidaatcache")
            expected_source = (PROJECT_ROOT / str(self.payload["source_video"])).resolve()
            if Path(str(candidate_payload.get("source_video", ""))).resolve() != expected_source:
                raise ValueError("Kandidaatcache hoort bij een andere video")
            self.candidate_frames = frames
        self.review_indices = review_indices
        self.review_position = 0
        if review_indices is not None:
            if not review_indices:
                raise ValueError("Geen zichtbare menselijke ballabels voor hercontrole")
            if any(not 0 <= index < len(self.annotations) for index in review_indices):
                raise ValueError("Hercontrole bevat een ongeldig annotatie-index")
            self.index = review_indices[0]
        else:
            self.index = next_required_review_index(self.annotations) or 0
        self.running = True
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None
        self.cursor = (0, 0)
        self.visibility = "visible"
        self.occlusion = "none"
        self.box: tuple[float, float, float, float] | None = None
        self.box_is_proposal = False
        self.click_box_size = 20.0
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.current_image: np.ndarray | None = None
        self.button_regions: list[tuple[tuple[int, int, int, int], str]] = []
        self._load_current_state()

    @property
    def current(self) -> dict[str, object]:
        return self.annotations[self.index]

    def run(self) -> None:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self.WIDTH, self.HEIGHT)
        cv2.setMouseCallback(self.WINDOW, self._mouse)
        try:
            while self.running:
                cv2.imshow(self.WINDOW, self._render())
                self._key(cv2.waitKeyEx(30))
        finally:
            save_review_manifest(self.payload, self.manifest_path)
            cv2.destroyWindow(self.WINDOW)

    def _image_path(self) -> Path:
        return self.manifest_path.parent / str(self.current["image"])

    def _load_current_state(self) -> None:
        self.current_image = cv2.imread(str(self._image_path()))
        if self.current_image is None:
            raise FileNotFoundError(self._image_path())
        raw_box = self.current.get("ball_box")
        self.box = None if raw_box is None else tuple(float(v) for v in raw_box)
        visibility = str(self.current.get("visibility", "unreviewed"))
        self.visibility = visibility if visibility != "unreviewed" else "visible"
        self.occlusion = str(self.current.get("occlusion", "none"))
        self.box_is_proposal = False
        frame_number = int(self.current["frame_number"])
        if (
            raw_box is None
            and visibility == "unreviewed"
            and frame_number < len(self.candidate_frames)
        ):
            height, width = self.current_image.shape[:2]
            self.box = proposed_candidate_box(
                self.candidate_frames[frame_number],
                image_width=width,
                image_height=height,
            )
            self.box_is_proposal = self.box is not None
        self.drag_start = self.drag_end = None

    def _fit_image(self) -> np.ndarray:
        assert self.current_image is not None
        height, width = self.current_image.shape[:2]
        self.scale = min(self.IMAGE_WIDTH / width, self.HEIGHT / height)
        resized = cv2.resize(
            self.current_image,
            (max(1, int(width * self.scale)), max(1, int(height * self.scale))),
        )
        self.offset_x = (self.IMAGE_WIDTH - resized.shape[1]) // 2
        self.offset_y = (self.HEIGHT - resized.shape[0]) // 2
        return resized

    def _render(self) -> np.ndarray:
        assert self.current_image is not None
        canvas = np.full((self.HEIGHT, self.WIDTH, 3), 24, dtype=np.uint8)
        displayed = self._fit_image()
        y2 = self.offset_y + displayed.shape[0]
        x2 = self.offset_x + displayed.shape[1]
        canvas[self.offset_y:y2, self.offset_x:x2] = displayed
        if self.box is not None:
            x1, y1, x2b, y2b = self.box
            cv2.rectangle(
                canvas,
                (int(x1 * self.scale) + self.offset_x, int(y1 * self.scale) + self.offset_y),
                (int(x2b * self.scale) + self.offset_x, int(y2b * self.scale) + self.offset_y),
                (0, 255, 0) if self.box_is_proposal else (0, 255, 255),
                3,
            )
        if self.drag_start and self.drag_end:
            cv2.rectangle(canvas, self.drag_start, self.drag_end, (0, 200, 255), 2)
        self._draw_magnifier(canvas)
        self._draw_sidebar(canvas)
        return canvas

    def _draw_magnifier(self, canvas: np.ndarray) -> None:
        assert self.current_image is not None
        if self.box is not None:
            image_x = int((self.box[0] + self.box[2]) / 2.0)
            image_y = int((self.box[1] + self.box[3]) / 2.0)
        else:
            image_x = int((self.cursor[0] - self.offset_x) / self.scale)
            image_y = int((self.cursor[1] - self.offset_y) / self.scale)
        h, w = self.current_image.shape[:2]
        if not (0 <= image_x < w and 0 <= image_y < h):
            return
        radius = 25
        crop_x1, crop_x2 = max(0, image_x - radius), min(w, image_x + radius)
        crop_y1, crop_y2 = max(0, image_y - radius), min(h, image_y + radius)
        crop = self.current_image[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return
        zoom_width, zoom_height = 320, 260
        zoom = cv2.resize(
            crop, (zoom_width, zoom_height), interpolation=cv2.INTER_NEAREST
        )
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            scale_x = zoom_width / (crop_x2 - crop_x1)
            scale_y = zoom_height / (crop_y2 - crop_y1)
            preview_box = (
                max(0, min(zoom_width - 1, int((x1 - crop_x1) * scale_x))),
                max(0, min(zoom_height - 1, int((y1 - crop_y1) * scale_y))),
                max(0, min(zoom_width - 1, int((x2 - crop_x1) * scale_x))),
                max(0, min(zoom_height - 1, int((y2 - crop_y1) * scale_y))),
            )
            cv2.rectangle(
                zoom,
                preview_box[:2],
                preview_box[2:],
                (0, 255, 0) if self.box_is_proposal else (0, 255, 255),
                2,
            )
        canvas[120:380, 1250:1570] = zoom
        cv2.drawMarker(canvas, (1410, 250), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)

    def _draw_sidebar(self, canvas: np.ndarray) -> None:
        x = 1240
        reviewed, total = review_progress(self.annotations)
        lines = [
            f"FRAME {self.current['frame_number']}  ({self.index + 1}/{total})",
            (
                f"Hercontrole: {self.review_position + 1}/{len(self.review_indices)}"
                if self.review_indices is not None
                else f"Voortgang: {reviewed}/{total}"
            ),
            "Sleep strak om bal/veeg; hoeft niet rond",
            "Enter: keuze + box opslaan",
            "+/-: box groter/kleiner",
            "A/D: vorige/volgende",
            "Q/Esc: sluiten (huidig vak niet opslaan)",
            f"Keuze: {self.visibility}",
            f"Occlusie: {self.occlusion}",
            self._box_size_label(),
            "Groene box = detectorvoorstel" if self.box_is_proposal else "Gele box = handmatig",
        ]
        for index, line in enumerate(lines):
            y = 42 + index * 34 if index < 2 else 405 + (index - 2) * 34
            color = (0, 255, 255) if index in (0, 1, 7, 8, 9) else (230, 230, 230)
            cv2.putText(canvas, line, (x, y), 0, 0.48, color, 1)
        self.button_regions = []
        buttons = (
            ("visible", "ZICHTBAAR + OPSLAAN", (40, 150, 40)),
            ("occluded", "BEDEKT + OPSLAAN", (20, 120, 180)),
            ("not_visible", "NIET ZICHTBAAR + OPSLAAN", (90, 90, 90)),
        )
        for index, (action, label, color) in enumerate(buttons):
            top = 710 + index * 58
            region = (1240, top, 1580, top + 48)
            self.button_regions.append((region, action))
            cv2.rectangle(canvas, region[:2], region[2:], color, -1)
            cv2.putText(canvas, label, (1252, top + 31), 0, 0.45, (255, 255, 255), 2)

    def _mouse(self, event: int, x: int, y: int, _flags: int, _data: object) -> None:
        self.cursor = (x, y)
        if x >= self.IMAGE_WIDTH:
            if event == cv2.EVENT_LBUTTONDOWN:
                for (x1, y1, x2, y2), action in self.button_regions:
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self._confirm_as(action)
                        return
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = self.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            self.drag_end = (x, y)
            assert self.current_image is not None
            h, w = self.current_image.shape[:2]
            candidate = image_box_from_drag(
                self.drag_start,
                self.drag_end,
                scale=self.scale,
                offset_x=self.offset_x,
                offset_y=self.offset_y,
                image_width=w,
                image_height=h,
            )
            if candidate is None:
                candidate = centered_image_box(
                    self.drag_end,
                    size=self.click_box_size,
                    scale=self.scale,
                    offset_x=self.offset_x,
                    offset_y=self.offset_y,
                    image_width=w,
                    image_height=h,
                )
            if candidate is not None:
                self.box = candidate
                self.box_is_proposal = False
                self.visibility = "visible"
                self.occlusion = "none"
            self.drag_start = self.drag_end = None

    def _key(self, key: int) -> None:
        if key in (27, ord("q"), ord("Q")):
            self.running = False
        elif key in (ord("v"), ord("V")):
            self.visibility = "visible"
            if self.occlusion == "player":
                self.occlusion = "none"
        elif key in (ord("o"), ord("O")):
            self.visibility = "occluded"
            self.occlusion = "player"
        elif key in (ord("x"), ord("X")):
            self.visibility = "not_visible"
            self.occlusion = "none"
            self.box = None
        elif key in (ord("c"), ord("C")):
            choices = ("none", "player", "shadow", "other")
            self.occlusion = choices[(choices.index(self.occlusion) + 1) % len(choices)]
        elif key in (ord("+"), ord("=")):
            self._resize_box(2.0)
        elif key in (ord("-"), ord("_")):
            self._resize_box(-2.0)
        elif key in (13, 10):
            self._confirm()
        elif key in (ord("a"), ord("A")):
            self._move(-1)
        elif key in (ord("d"), ord("D")):
            self._move(1)

    def _confirm(self) -> None:
        try:
            self.annotations[self.index] = confirm_ball_annotation(
                self.current,
                visibility=self.visibility,
                box=self.box,
                occlusion=self.occlusion,
            )
        except ValueError as error:
            print(f"Nog niet opgeslagen: {error}")
            return
        save_review_manifest(self.payload, self.manifest_path)
        if self.review_indices is not None:
            self.review_position += 1
            if self.review_position >= len(self.review_indices):
                self.running = False
            else:
                self.index = self.review_indices[self.review_position]
                self._load_current_state()
            return
        next_index = next_required_review_index(self.annotations, after=self.index)
        if next_index is None:
            self.running = False
        else:
            self.index = next_index
            self._load_current_state()

    def _confirm_as(self, visibility: str) -> None:
        self.visibility = visibility
        if visibility == "occluded":
            self.occlusion = "player"
        elif visibility == "not_visible":
            self.occlusion = "none"
            self.box = None
        else:
            self.occlusion = "none"
        self._confirm()

    def _resize_box(self, delta: float) -> None:
        self.click_box_size = min(80.0, max(4.0, self.click_box_size + delta))
        if self.box is None:
            return
        center_x = (self.box[0] + self.box[2]) / 2.0
        center_y = (self.box[1] + self.box[3]) / 2.0
        assert self.current_image is not None
        h, w = self.current_image.shape[:2]
        box_width = min(80.0, max(4.0, self.box[2] - self.box[0] + delta))
        box_height = min(80.0, max(4.0, self.box[3] - self.box[1] + delta))
        half_width = box_width / 2.0
        half_height = box_height / 2.0
        self.box = (
            max(0.0, center_x - half_width),
            max(0.0, center_y - half_height),
            min(float(w), center_x + half_width),
            min(float(h), center_y + half_height),
        )
        self.box_is_proposal = False

    def _box_size_label(self) -> str:
        if self.box is None:
            return f"Nieuwe box: {self.click_box_size:.0f}x{self.click_box_size:.0f}px"
        width = self.box[2] - self.box[0]
        height = self.box[3] - self.box[1]
        return f"Boxmaat: {width:.0f}x{height:.0f}px"

    def _move(self, step: int) -> None:
        if self.review_indices is not None:
            self.review_position = min(
                max(0, self.review_position + step), len(self.review_indices) - 1
            )
            self.index = self.review_indices[self.review_position]
            self._load_current_state()
            return
        self.index = min(max(0, self.index + step), len(self.annotations) - 1)
        self._load_current_state()


def main() -> None:
    parser = argparse.ArgumentParser(description="Review actieve-bal-ground-truth.")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--no-candidate-prefill", action="store_true")
    parser.add_argument(
        "--recheck-visible",
        action="store_true",
        help="Hercontroleer alleen bestaande menselijk beoordeelde zichtbare ballen.",
    )
    parser.add_argument(
        "--propagate-from-reviewed",
        action="store_true",
        help="Volg vanaf het laatste menselijke zichtbare anker en sla AI-concepten op.",
    )
    args = parser.parse_args()
    if args.propagate_from_reviewed:
        payload = json.loads(args.annotations.read_text(encoding="utf-8"))
        drafted = propagate_from_last_reviewed(payload, args.annotations)
        print(f"{drafted} AI-conceptframes gevolgd; menselijke labels zijn niet gewijzigd.")
        return
    candidate_cache = args.candidate_cache
    if candidate_cache is None and not args.no_candidate_prefill:
        manifest = json.loads(args.annotations.read_text(encoding="utf-8"))
        video_stem = Path(str(manifest["source_video"])).stem
        inferred = PROJECT_ROOT / "output" / "ball" / f"{video_stem}_ball_candidates.json"
        if inferred.exists():
            candidate_cache = inferred
    review_indices = None
    if args.recheck_visible:
        manifest = json.loads(args.annotations.read_text(encoding="utf-8"))
        review_indices = visible_human_review_indices(manifest["annotations"])
    BallGroundTruthReviewApp(
        args.annotations,
        candidate_cache,
        review_indices=review_indices,
    ).run()


if __name__ == "__main__":
    main()
