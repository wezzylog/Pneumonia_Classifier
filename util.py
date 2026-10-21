import base64
import os

import numpy as np
import streamlit as st
from PIL import Image, ImageOps


def set_background(image_file):
    """Set a full-page background image. Silently skipped if the file is missing."""
    if not os.path.exists(image_file):
        return
    with open(image_file, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64});
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def classify(image, model, class_names):
    """Return (class_name, confidence) for a PIL image.

    Preprocessing must match train.py: 224x224 RGB, scaled to [-1, 1].
    """
    image = ImageOps.fit(image.convert('RGB'), (224, 224), Image.Resampling.LANCZOS)
    arr = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    probs = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx])