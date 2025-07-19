# model_setup.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Или обучи на своих данных, если нужно точнее
model.fuse()  # Ускорение
