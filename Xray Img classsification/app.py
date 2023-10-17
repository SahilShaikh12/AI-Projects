import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2

# Load the YOLO model
model_path = r'C:\Users\user1\PycharmProjects\pythonProject\venv\runs\classify\train8\weights\best.pt'
yolo_model = YOLO(model_path)

# Streamlit app
st.title("YOLO Image Classifier")

uploaded_file = st.file_uploader("Choose an image to upload", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Perform prediction when a file is uploaded
    if st.button("Predict"):
        # Read the image and make predictions
        img = cv2.imread(uploaded_file.name)
        results = yolo_model(img)

        # Get the class names and probabilities
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()

        # Get the index of the class with the highest probability
        highest_prob_index = np.argmax(probs)
        highest_prob_class = names_dict[highest_prob_index]
        highest_prob_percentage = probs[highest_prob_index] * 100

        # Display the prediction results
        st.write("Prediction Results:")
        st.write(f"Class with the highest probability: {highest_prob_class}")
        st.write(f"Percentage accuracy: {highest_prob_percentage:.2f}%")
