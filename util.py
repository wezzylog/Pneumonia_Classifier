import base64
import os

import numpy as np
import streamlit as st
from PIL import Image, ImageOps


def set_background(image_file):
    """Set a local image as the Streamlit app background.
    Silently skips if the file is missing so the app still runs."""
    if not os.path.exists(image_file):
        return
    with open(image_file, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def classify(image, model, class_names, size=(224, 224)):
    """Return (class_name, confidence) for a PIL image.

    Uses the Teachable Machine preprocessing: resize/crop to 224x224,
    then scale pixels to [-1, 1]. Change `size` and the scaling if your
    model was trained differently.
    """
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    arr = np.asarray(image).astype(np.float32)
    arr = (arr / 127.5) - 1

    batch = np.expand_dims(arr, axis=0)  # shape (1, 224, 224, 3)
    prediction = model.predict(batch, verbose=0)

    index = int(np.argmax(prediction[0]))
    return class_names[index], float(prediction[0][index])
