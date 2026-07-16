from pathlib import Path

import cv2
from rfdetr import RFDETRBase

PROJECT_DIR = Path.home() / "football-ai"
VIDEO_PATH = PROJECT_DIR / "videos" / "Test4k.mp4"

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video niet gevonden: {VIDEO_PATH}")

print(f"Video openen: {VIDEO_PATH}")

cap = cv2.VideoCapture(str(VIDEO_PATH))
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Het eerste videoframe kon niet worden gelezen.")

print(f"Frame ingelezen: {frame.shape[1]} × {frame.shape[0]}")
print("RF-DETR-model laden...")

model = RFDETRBase()
results = model.predict(frame, threshold=0.3)

print("Detectie voltooid:")
print(results)
