import cv2
import numpy as np

# Define the color range for black
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 30])  # Increase the value range for different lighting conditions

# Initialize the webcam (usually 0 for built-in webcams)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the black color
    mask = cv2.inRange(hsv_frame, lower_black, upper_black)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around the detected contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the original image with rectangles
    cv2.imshow('Black Color Detection with Rectangles', frame)

    # Exit the loop and close the window when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
