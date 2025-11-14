#Streamlit_app_V2

import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import  preprocess_input, decode_predictions

load_dotenv()
# Load the MobileNetV2 model with ImageNet weights

MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS")

model = tf.keras.models.load_model(f'weights/{MODEL_WEIGHTS}')


def preprocess_image(img, target_size=(224, 224)):
    """
    Preprocess an image for MobileNetV2 model.

    Args:
    img (PIL.Image.Image): Image file.

    Returns:
    numpy.ndarray: Preprocessed image array.
    """
    # Resize the image
    img = img.resize(target_size)

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand dimensions to match the shape the model expects: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image array
    result_img_array = preprocess_input(img_array)

    return result_img_array

# Streamlit app
st.title("Image Classification with MobileNet V2")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    img = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    preprocessed_image = preprocess_image(img)

    # Prediction
    prediction = model.predict(preprocessed_image)

    # Decode the prediction
    decoded_prediction = decode_predictions(prediction, top=5)[0]

    # Display the prediction
    st.write("Top 5 Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_prediction):
        st.write(f"{i+1}: {label} ({score:.2f})")
