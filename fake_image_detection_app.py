# Import necessary libraries
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Load a pre-trained GAN model (you should have a trained model for this)
gan_model = keras.models.load_model('gan_model.h5')

# Define a function to preprocess and classify an image
def classify_image(image):
    # Resize and preprocess the image
    image = np.array(image)
    image = tf.image.resize(image, (128, 128)) / 255.0  # Resize to match GAN input size
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict using the GAN model
    prediction = gan_model.predict(image)

    # Determine if the image is fake or real based on the prediction
    if prediction[0][0] > 0.5:
        return "Fake Image"
    else:
        return "Real Image"

# Create a Streamlit web app
st.title("Fake Image Detection")

# Upload an image for classification
st.write("Upload an image for classification")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Classify the image
    result = classify_image(image)
    st.write(f"Classification result: {result}")
