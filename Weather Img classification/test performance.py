from ultralytics import YOLO

import numpy as np


model = YOLO(r'C:\Users\user1\PycharmProjects\pythonProject\venv\runs\classify\train2\weights\best.pt')  # load a custom model

results = model(r'C:\Users\user1\Downloads\weather\train\Sunny\3721830.jpg')  # predict on an image

names_dict = results[0].names

probs = results[0].probs.data.tolist()

print(names_dict)
print(probs)

print(names_dict[np.argmax(probs)])