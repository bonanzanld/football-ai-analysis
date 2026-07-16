from pathlib import Path
import time

import cv2
from rfdetr import RFDETRMedium

PROJECT_DIR = Path.home() / "football-ai"
VIDEO_PATH = PROJECT_DIR / "videos" / "Test4k.mp4"
OUTPUT_PATH = PROJECT_DIR / "output" / "Test4k_detected_10s.mp4"

START_SECONDS = 0
DURATION_SECONDS = 10
THRESHOLD = 0.25

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Kan video niet openen: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_POS_MSEC, START_SECONDS * 1000)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

model = RFDETRMedium()
max_frames = int(DURATION_SECONDS * fps)
ball_frames = 0
person_detections = 0

start = time.time()

for frame_number in range(max_frames):
    ret, frame = cap.read()
    if not ret:
        break

    detections = model.predict(frame, threshold=THRESHOLD)

    for box, confidence, class_name in zip(
        detections.xyxy,
        detections.confidence,
        detections.data["class_name"],
    ):
        if class_name not in {"person", "sports ball"}:
            continue

        x1, y1, x2, y2 = map(int, box)

        if class_name == "person":
            person_detections += 1
            label = f"person {confidence:.2f}"
            color = (0, 255, 0)
        else:
            ball_frames += 1
            label = f"ball? {confidence:.2f}"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 7, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"Frame {frame_number + 1}/{max_frames}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    writer.write(frame)

    if (frame_number + 1) % 30 == 0:
        print(f"{frame_number + 1}/{max_frames} frames verwerkt")

cap.release()
writer.release()

elapsed = time.time() - start

print(f"Klaar: {OUTPUT_PATH}")
print(f"Verwerkingstijd: {elapsed:.1f} seconden")
print(f"Person-detecties totaal: {person_detections}")
print(f"Ball-detecties totaal: {ball_frames}")
