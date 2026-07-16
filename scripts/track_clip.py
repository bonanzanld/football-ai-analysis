from pathlib import Path
import time

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium

PROJECT_DIR = Path.home() / "football-ai"
VIDEO_PATH = PROJECT_DIR / "videos" / "Test4k.mp4"
OUTPUT_PATH = PROJECT_DIR / "output" / "Test4k_tracked_10s.mp4"

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

tracker = sv.ByteTrack(
    frame_rate=round(fps),
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
)

max_frames = int(DURATION_SECONDS * fps)

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(
    text_scale=0.45,
    text_thickness=1,
    text_padding=3,
)

start_time = time.time()

for frame_number in range(max_frames):
    ret, frame = cap.read()
    if not ret:
        break

    detections = model.predict(frame, threshold=THRESHOLD)

    class_names = detections.data["class_name"]

    keep_mask = np.isin(class_names, ["person", "sports ball"])
    detections = detections[keep_mask]

    detections = tracker.update_with_detections(detections)

    labels = []

    for class_name, confidence, tracker_id in zip(
        detections.data["class_name"],
        detections.confidence,
        detections.tracker_id,
    ):
        if class_name == "person":
            labels.append(f"P#{tracker_id} {confidence:.2f}")
        else:
            labels.append(f"B?#{tracker_id} {confidence:.2f}")

    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections,
    )

    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels,
    )

    cv2.putText(
        annotated,
        f"Frame {frame_number + 1}/{max_frames}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    writer.write(annotated)

    if (frame_number + 1) % 30 == 0:
        print(f"{frame_number + 1}/{max_frames} frames verwerkt")

cap.release()
writer.release()

elapsed = time.time() - start_time

print(f"Klaar: {OUTPUT_PATH}")
print(f"Verwerkingstijd: {elapsed:.1f} seconden")
