import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# Welcome to Kidney Disease Detection from CT/MRI Scans Portal

This site is based on CNN based ML Model.

Accuracy of the model is around 95%.
"""

import streamlit as st
import tensorflow as tf
import numpy as np
# import cv2
from PIL import Image
import requests
from io import BytesIO

# Rebuild the model architecture
def build_model():
    model = tf.keras.Sequential([
        # Add layers here to match your original model architecture
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3), kernel_regularizer=tf.keras.regularizers.l2(0.0001)), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation=tf.nn.softmax)])
    return model

# Initialize and load weights into the model
model = build_model()
model.load_weights('/workspaces/kidney-detect/First_Model_V3.h5')  # Assuming weights are saved separately

# Define the label to class name mapping
label_to_class_name = {0: 'Cyst', 1: 'Normal', 2: 'Stone', 3: 'Tumor'}

# Streamlit app
st.title("CT Kidney Image Classifier")

# URL input
url = st.text_input("Enter the image URL:")

if url:
    try:
        # Read image from URL
        response = requests.get(url)
        response.raise_for_status() 
        img = Image.open(BytesIO(response.content))

        # Display the image
        st.image(img, caption='Input Image', use_column_width=True)

        # Convert image to array
        img = np.array(img)

        # Resize image to the expected input shape for the model
        img_resized = tf.image.resize(img, (150, 150))

        # Predict the class
        yhat = model.predict(np.expand_dims(img_resized / 255.0, axis=0))
        max_index = np.argmax(yhat)
        predicted_class = label_to_class_name[max_index]

        # Display the predicted class
        st.write(f"Predicted Class: {predicted_class}")

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the image: {e}")
    except UnidentifiedImageError:
        st.error("Error: The URL does not point to a valid image file.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
