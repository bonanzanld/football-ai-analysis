from pathlib import Path

import cv2
from rfdetr import RFDETRMedium

PROJECT_DIR = Path.home() / "football-ai"
VIDEO_PATH = PROJECT_DIR / "videos" / "Test4k.mp4"
OUTPUT_PATH = PROJECT_DIR / "output" / "first_frame_detected.jpg"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO_PATH))
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Het eerste videoframe kon niet worden gelezen.")

print("Model laden...")
model = RFDETRMedium()

print("Detectie uitvoeren...")
detections = model.predict(frame, threshold=0.30)

for box, confidence, class_name in zip(
    detections.xyxy,
    detections.confidence,
    detections.data["class_name"],
):
    x1, y1, x2, y2 = map(int, box)

    if class_name == "person":
        label = f"person {confidence:.2f}"
        thickness = 2
    elif class_name == "sports ball":
        label = f"ball? {confidence:.2f}"
        thickness = 3
    else:
        continue

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)

    text_y = max(y1 - 8, 20)
    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

cv2.imwrite(str(OUTPUT_PATH), frame)

print(f"Klaar: {OUTPUT_PATH}")
print(f"Aantal detecties: {len(detections)}")
