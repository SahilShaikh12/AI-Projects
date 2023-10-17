from ultralytics import YOLO

import numpy as np


model = YOLO(r'C:\Users\user1\PycharmProjects\pythonProject\venv\runs\classify\train8\weights\best.pt')  # load a custom model

results = model(r'C:\Users\user1\OneDrive\Desktop\images.jpeg')  # predict on an image

names_dict = results[0].names

probs = results[0].probs.data.tolist()

# Get the index of the class with the highest probability
highest_prob_index = np.argmax(probs)

# Print the class with the highest probability and its percentage accuracy
highest_prob_class = names_dict[highest_prob_index]
highest_prob_percentage = probs[highest_prob_index] * 100
print(f"Class with the highest probability: {highest_prob_class}")
print(f"Percentage accuracy: {highest_prob_percentage:.2f}%")

print(names_dict)
print(probs)

print(names_dict[np.argmax(probs)])