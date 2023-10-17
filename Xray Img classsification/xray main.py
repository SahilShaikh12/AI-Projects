import os
import cv2
from ultralytics import YOLO

# Start training from a pretrained *.pt model
model=('yolov8n-cls.pt')
yolo=YOLO(model)
# load dataset from direct.
yolo.train(data=r"C:\Users\user1\OneDrive\Desktop\xray data", epochs=2 , imgsz=64)