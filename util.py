import base64
import mimetypes
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

IMG_SIZE = (224, 224)  # keep in sync with train.py


@st.cache_data(show_spinner=False)
def _encode_image(path: str) -> tuple[str, str]:
    """Read and base64-encode a file once; cached across Streamlit reruns."""
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return base64.b64encode(Path(path).read_bytes()).decode(), mime


def set_background(image_file: str) -> None:
    """Set a full-page background image. Silently skipped if the file is missing."""
    if not Path(image_file).is_file():
        return
    b64, mime = _encode_image(image_file)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:{mime};base64,{b64});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def preprocess(image: Image.Image) -> np.ndarray:
    """224x224 RGB, scaled to [-1, 1]. Must match train.py."""
    image = ImageOps.exif_transpose(image)  # respect phone-camera rotation
    image = ImageOps.fit(image.convert("RGB"), IMG_SIZE, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 127.5 - 1.0


def predict_probs(image: Image.Image, model, class_names: list[str]) -> np.ndarray:
    probs = model.predict(preprocess(image)[None, ...], verbose=0)[0]
    if len(probs) != len(class_names):
        raise ValueError(
            f"Model outputs {len(probs)} classes but {len(class_names)} names were given."
        )
    return probs


def classify(image: Image.Image, model, class_names: list[str]) -> tuple[str, float]:
    """Return (class_name, confidence) for a PIL image."""
    probs = predict_probs(image, model, class_names)
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx])


def classify_top_k(image: Image.Image, model, class_names: list[str], k: int = 3):
    """Return the k most likely (class_name, confidence) pairs, best first."""
    probs = predict_probs(image, model, class_names)
    top = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in top]