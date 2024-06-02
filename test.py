from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
import cv2

model = YOLO("yolov8s.pt")
results = model.predict(source="0", show=True)

print(results)