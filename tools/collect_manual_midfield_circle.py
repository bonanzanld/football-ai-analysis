from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from collect_manual_8v8_endline import EndLineCollector, PROJECT_ROOT


class MidfieldCircleCollector(EndLineCollector):
    WINDOW = "Football AI - middenanker via middencirkel"

    def __init__(self, video: Path, start: float, minimum: float, maximum: float, draft: Path) -> None:
        super().__init__(video, start, minimum, maximum, "middle")
        self.required_points = 7
        self.draft = draft
        self.status = "Klik zeven verspreide punten op dezelfde witte middencirkelboog."

    def _payload(self, complete: bool) -> dict:
        return {
            "schema_version": 1,
            "video_name": self.video.name,
            "frame_number": int(round(self.time * self.fps)),
            "time_seconds": self.time,
            "role": "11v11_center_circle_midfield_bridge",
            "official_radius_m": 9.15,
            "points": [list(map(float, point)) for point in self.points],
            "complete": complete,
            "fit_status": "pending_camera_constrained_fit",
            "provenance": "human_reviewed" if complete else "human_review_in_progress",
        }

    def _save_draft(self) -> None:
        self.draft.parent.mkdir(parents=True, exist_ok=True)
        self.draft.write_text(
            json.dumps(self._payload(len(self.points) == self.required_points), indent=2),
            encoding="utf-8",
        )

    def _mouse(self, event, x, y, flags, data):
        before = len(self.points)
        super()._mouse(event, x, y, flags, data)
        if len(self.points) != before:
            self._save_draft()

    def run(self) -> dict:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self.WIDTH, self.HEIGHT)
        cv2.setMouseCallback(self.WINDOW, self._mouse)
        try:
            while not self.done:
                cv2.imshow(self.WINDOW, self._render())
                key = cv2.waitKeyEx(30)
                if key in (27, ord("q"), ord("Q")):
                    raise KeyboardInterrupt
                if key in (ord("u"), ord("U"), 8, 127) and self.points:
                    self.points.pop()
                    self._save_draft()
                elif key in (ord(","), ord("<")) and not self.points:
                    self._shift(-1.0)
                elif key in (ord("."), ord(">")) and not self.points:
                    self._shift(1.0)
                elif key in (ord("+"), ord("=")):
                    self.zoom = min(8.0, self.zoom * 1.25)
                elif key in (ord("-"), ord("_")):
                    self.zoom = max(1.0, self.zoom / 1.25)
                elif key == ord("0"):
                    self.zoom, self.center = 1.0, None
                elif key in (2424832, 65361, 63234, 81, ord("a"), ord("A")):
                    self._pan(-0.16, 0.0)
                elif key in (2555904, 65363, 63235, 83, ord("d"), ord("D")):
                    self._pan(0.16, 0.0)
                elif key in (2490368, 65362, 63232, 82, ord("w"), ord("W")):
                    self._pan(0.0, -0.16)
                elif key in (2621440, 65364, 63233, 84, ord("s"), ord("S")):
                    self._pan(0.0, 0.16)
                elif key in (10, 13) and len(self.points) == self.required_points:
                    self.done = True
        finally:
            self.capture.release()
            cv2.destroyWindow(self.WINDOW)
        result = self._payload(True)
        self.draft.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _render(self):
        canvas = super()._render()
        cv2.rectangle(canvas, (0, 0), (self.PANEL - 1, self.HEIGHT), (24, 24, 24), -1)
        lines = (
            "MIDDENANKER: 11v11-MIDDENCIRKEL", "",
            "Klik 7 verspreide punten op de", "WITTE BOOG van dezelfde middencirkel.", "",
            "De gele lijn is alleen de KLIKBOOG.", "Dit is bewust nog GEEN ellips.", "",
            f"Tijd: {self.time:.1f}s", f"Punten: {len(self.points)}/{self.required_points}", "",
            ",/. ander moment (voor eerste klik)", "Muiswiel of +/- zoom",
            "Pijltjes of W/A/S/D bewegen", "U terug | Enter opslaan", "Esc afbreken",
        )
        for index, line in enumerate(lines):
            color = (0, 230, 255) if index in (0, 2, 3, 5, 6) else (235, 235, 235)
            cv2.putText(
                canvas, line, (18, 36 + index * 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.53, color, 2 if index == 0 else 1, cv2.LINE_AA,
            )
        if len(self.points) >= 2:
            x0, y0, scale, ox, oy = self.mapping
            displayed = np.asarray(
                [(round(ox + (x - x0) * scale), round(oy + (y - y0) * scale)) for x, y in self.points],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [displayed], False, (0, 255, 255), 3, cv2.LINE_AA)
        return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Leg de zichtbare 11v11-middencirkelboog vast.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--format", choices=("6v6", "8v8", "11v11"), default="8v8")
    parser.add_argument("--start", type=float, default=849.0)
    parser.add_argument("--minimum", type=float, default=838.0)
    parser.add_argument("--maximum", type=float, default=860.0)
    args = parser.parse_args()
    video = PROJECT_ROOT / "videos" / args.video
    output = PROJECT_ROOT / "output" / "pitch_bootstrap" / f"{video.stem}_{args.format}_manual_midfield_circle.json"
    draft = output.with_name(f"{output.stem}_draft.json")
    result = MidfieldCircleCollector(video, args.start, args.minimum, args.maximum, draft).run()
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Handmatige middencirkelboog opgeslagen: {output}")


if __name__ == "__main__":
    main()
