# model_setup.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.fuse()  # Ускорение
