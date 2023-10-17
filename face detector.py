import cv2
import os
import numpy as np

# Load the pre-trained Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image/webcam
#image_path = r'C:\Users\user1\OneDrive\Pictures\Camera Roll\WIN_20230929_16_20_28_Pro.jpg'  # Replace with the actual image path
cap= cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the image with rectangles around faces
        cv2.imshow('Face Detection', frame)

    # Exit the loop and close the window when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
#cv2.waitKey(0)
#cv2.destroyAllWindows()
